// SPDX-License-Identifier: Apache-2.0
#include "qwen3.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>

#include "nvtx_helper.h"
#include "ort_session.h"
#include "tokenizer.h"
#include "unicode_regex.h"
#include <nlohmann/json.hpp>

namespace din::asr::qwen3::detail
{
inline int ResolveChunkSeconds(int seconds, bool aligned)
{
    const int limit = aligned ? 180 : 1200;
    if (seconds == 0)
        return limit;
    // A target inside the search radius can repeatedly split silence into single samples.
    if (seconds <= 5 || seconds > limit)
        throw std::invalid_argument(aligned ? "max-chunk-seconds must be 0 (auto) or 6..180 with alignment"
                                            : "max-chunk-seconds must be 0 (auto) or 6..1200");
    return seconds;
}

struct AudioChunk
{
    size_t begin;
    size_t end;
};

inline float WindowEnergy(std::span<const float> samples, size_t begin)
{
    // Independent FP32 reductions avoid cancellation drift across quiet windows.
    std::array<float, 8> sums{};
    for (size_t i = 0; i < 1600; i += 8)
        for (size_t j = 0; j < 8; ++j)
            sums[j] += std::abs(samples[begin + i + j]);
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
}

// Qwen3-ASR split_audio_into_chunks: quietest 100 ms within +/-5 seconds,
// then the quietest sample inside that window. Keep the first minimum on ties.
inline std::vector<AudioChunk> SplitAudio(std::span<const float> samples, size_t target_seconds = 1200)
{
    const size_t target = target_seconds * 16000;
    constexpr size_t expand = 5 * 16000, window = 1600;
    std::vector<AudioChunk> chunks;
    size_t start = 0;
    while (samples.size() - start > target)
    {
        const auto cut = start + target;
        const auto left = cut > expand ? std::max(start, cut - expand) : start;
        const auto right = std::min(samples.size(), cut + expand);
        size_t boundary = cut;
        if (right - left > window)
        {
            float best = WindowEnergy(samples, left);
            size_t minimum = left;
            for (size_t i = left + 1; i + window <= right; ++i)
            {
                const float sum = WindowEnergy(samples, i);
                if (sum < best)
                {
                    best = sum;
                    minimum = i;
                }
            }
            boundary = minimum;
            for (size_t i = minimum + 1; i < minimum + window; ++i)
                if (std::abs(samples[i]) < std::abs(samples[boundary]))
                    boundary = i;
        }
        boundary = std::clamp(boundary, start + 1, samples.size());
        chunks.push_back({start, boundary});
        start = boundary;
    }
    if (start < samples.size())
        chunks.push_back({start, samples.size()});
    return chunks;
}
}  // namespace din::asr::qwen3::detail

namespace din::asr::qwen3::detail
{
std::string Trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \r\n\t");
    if (first == std::string::npos)
        return {};
    return text.substr(first, text.find_last_not_of(" \r\n\t") - first + 1);
}

// Official detect_and_fix_repetitions operates on Unicode characters, not UTF-8 bytes.
static std::string FixRepetitions(const std::string& text)
{
    std::vector<std::string_view> chars, filtered;
    for (size_t i = 0; i < text.size();)
    {
        const auto c = static_cast<unsigned char>(text[i]);
        const size_t length = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
        chars.emplace_back(text.data() + i, std::min(length, text.size() - i));
        i += length;
    }
    for (size_t i = 0; i < chars.size();)
    {
        size_t end = i + 1;
        while (end < chars.size() && chars[end] == chars[i])
            ++end;
        filtered.insert(filtered.end(), chars.begin() + i, chars.begin() + (end - i > 20 ? i + 1 : end));
        i = end;
    }
    std::string result;
    for (size_t i = 0; i < filtered.size();)
    {
        size_t advance = 1, keep = 1;
        if (filtered.size() - i >= 40)
            for (size_t length = 1; length <= 20 && i + length * 20 <= filtered.size(); ++length)
            {
                size_t end = i + length;
                while (end + length <= filtered.size() &&
                       std::equal(filtered.begin() + i, filtered.begin() + i + length, filtered.begin() + end))
                    end += length;
                if ((end - i) / length >= 20)
                {
                    keep = length;
                    advance = end - i;
                    break;
                }
            }
        for (size_t j = 0; j < keep; ++j)
            result += filtered[i + j];
        i += advance;
    }
    return result;
}

std::pair<std::string, std::string> ParseOutput(const std::string& raw, const std::string& forced_language = {})
{
    const auto text = FixRepetitions(Trim(raw));
    if (text.empty())
        return {};
    if (!forced_language.empty())
        return {forced_language, text};
    const auto marker = text.find("<asr_text>");
    if (marker == std::string::npos)
        return {{}, Trim(text)};
    auto meta = text.substr(0, marker);
    std::transform(meta.begin(), meta.end(), meta.begin(),
                   [](unsigned char c)
                   {
                       return std::tolower(c);
                   });
    const auto transcript = Trim(text.substr(marker + 10));
    if (meta.find("language none") != std::string::npos)
        return {{}, transcript};
    std::istringstream lines(meta);
    std::string line;
    while (std::getline(lines, line))
    {
        line = Trim(line);
        if (line.starts_with("language "))
        {
            auto language = Trim(line.substr(9));
            if (!language.empty())
                language[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(language[0])));
            return {language, transcript};
        }
    }
    return {{}, transcript};
}

std::string LowerLanguage(const std::string& language)
{
    auto result = Trim(language);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c)
                   {
                       return static_cast<char>(std::tolower(c));
                   });
    return result;
}

std::string AsrLanguage(const std::string& language, const nlohmann::json& languages)
{
    const auto hint = LowerLanguage(language);
    for (const auto& [key, value] : languages.items())
        if (LowerLanguage(key) == hint)
            return key;
    throw std::invalid_argument("Unknown ASR language hint; use an upstream language code/name or auto");
}

std::vector<std::string> AlignmentUnits(const std::string& text, const std::string& language)
{
    static constexpr std::pair<std::string_view, std::string_view> languages[] = {
        {"zh", "chinese"}, {"yue", "cantonese"}, {"en", "english"}, {"de", "german"},
        {"es", "spanish"}, {"fr", "french"},     {"it", "italian"}, {"pt", "portuguese"},
        {"ru", "russian"}, {"ko", "korean"},     {"ja", "japanese"}};
    const auto name = LowerLanguage(language);
    const auto found = std::find_if(std::begin(languages), std::end(languages),
                                    [&](const auto& entry)
                                    {
                                        return name == entry.first || name == entry.second;
                                    });
    if (found == std::end(languages))
        throw std::invalid_argument("Forced alignment supports zh, yue, en, de, es, fr, it, pt, ru, ko and ja");

    // HF keeps Unicode letters/numbers and ASCII apostrophes, dropping punctuation and marks.
    static const din::io::UnicodeRegex kept(R"([\p{L}\p{N}'\s\x{1c}-\x{1f}]+)");
    std::string cleaned;
    for (const auto& part : kept.FindAll(text))
        cleaned += part;
    static const std::string cjk = R"(\x{4e00}-\x{9fff}\x{3400}-\x{4dbf}\x{20000}-\x{2a6df}\x{2a700}-\x{2b73f})"
                                   R"(\x{2b740}-\x{2b81f}\x{2b820}-\x{2ceaf}\x{f900}-\x{faff}\x{2f800}-\x{2fa1f})";
    static const din::io::UnicodeRegex words("[" + cjk + "]|[^" + cjk + R"(\s\x{1c}-\x{1f}]+)");
    // HF's unscored Korean LTokenizer splits on whitespace before cleaning each unit.
    static const din::io::UnicodeRegex korean(R"([^\s\x{1c}-\x{1f}]+)");
    // Japanese character timestamps avoid a separate Nagisa word-segmentation runtime.
    static const std::string kana = R"(\x{3040}-\x{30ff}\x{31f0}-\x{31ff}\x{ff66}-\x{ff9f})";
    static const din::io::UnicodeRegex japanese("[" + cjk + kana + "]|[^" + cjk + kana + R"(\s\x{1c}-\x{1f}]+)");
    if (found->first == "ja")
        return japanese.FindAll(cleaned);
    if (found->first == "ko")
        return korean.FindAll(cleaned);
    return words.FindAll(cleaned);
}

// HF's nondecreasing subsequence repair, including its tie rules.
std::vector<int> FixTimestamps(const std::vector<int>& data)
{
    const int n = static_cast<int>(data.size());
    if (n == 0)
        return {};
    std::vector<int> dp(n, 1), parent(n, -1), result = data;
    std::vector<bool> normal(n, false);
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (data[j] <= data[i] && dp[j] + 1 > dp[i])
            {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
    int i = static_cast<int>(std::max_element(dp.begin(), dp.end()) - dp.begin());
    for (; i >= 0; i = parent[i])
        normal[i] = true;
    for (int begin = 0; begin < n;)
    {
        if (normal[begin])
        {
            ++begin;
            continue;
        }
        int end = begin;
        while (end < n && !normal[end])
            ++end;
        for (int k = begin; k < end; ++k)
        {
            if (begin == 0)
                result[k] = result[end];
            else if (end == n)
                result[k] = result[begin - 1];
            else if (end - begin <= 2)
                result[k] = k - begin + 1 <= end - k ? result[begin - 1] : result[end];
            else
                result[k] = static_cast<int>(result[begin - 1] + (result[end] - result[begin - 1]) *
                                                                     float(k - begin + 1) / (end - begin + 1));
        }
        begin = end;
    }
    return result;
}
}  // namespace din::asr::qwen3::detail

namespace din::asr::qwen3
{
namespace
{
using Json = nlohmann::json;
using BF16 = Ort::BFloat16_t;
using din::common::OrtRunner;
template <class T>
using Buffer = din::common::TensorBuffer<T>;
class FloatBuffer
{
    using Storage = std::variant<Buffer<float>, Buffer<Ort::Float16_t>, Buffer<BF16>>;
    Storage buffer_;

    static Storage Make(OrtRunner& runner, const std::vector<int64_t>& shape, ONNXTensorElementDataType dtype,
                        bool disable_uma)
    {
        switch (dtype)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return Storage(std::in_place_type<Buffer<float>>, runner, shape, true, disable_uma);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return Storage(std::in_place_type<Buffer<Ort::Float16_t>>, runner, shape, true, disable_uma);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return Storage(std::in_place_type<Buffer<BF16>>, runner, shape, true, disable_uma);
        default:
            throw std::runtime_error("Unsupported Qwen3 tensor precision");
        }
    }

public:
    FloatBuffer(OrtRunner& runner, const std::vector<int64_t>& shape, ONNXTensorElementDataType dtype,
                bool disable_uma = false)
        : buffer_(Make(runner, shape, dtype, disable_uma))
    {
    }

    template <class F>
    void WithHost(F&& fill)
    {
        std::visit(
            [&](auto& buffer)
            {
                fill(buffer.HostData());
            },
            buffer_);
    }
    void Fill(float value)
    {
        std::visit(
            [&](auto& buffer)
            {
                using T = std::remove_pointer_t<decltype(buffer.HostData())>;
                buffer.Fill(T(value));
            },
            buffer_);
    }
    Ort::Value& BindingValue()
    {
        return std::visit(
            [](auto& buffer) -> Ort::Value&
            {
                return buffer.BindingValue();
            },
            buffer_);
    }
    void CopyAsyncToDevice()
    {
        std::visit(
            [](auto& buffer)
            {
                buffer.CopyAsyncToDevice();
            },
            buffer_);
    }
    void UploadAndWait()
    {
        std::visit(
            [](auto& buffer)
            {
                buffer.CopyAsyncToDeviceWithNotification().Sync();
            },
            buffer_);
    }
};

size_t ElementBytes(const Ort::Value& value)
{
    return value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 2;
}

constexpr int kRate = 16000;
constexpr int kMaxSamples = 1205 * kRate;
constexpr int kPrefillBlock = 512;

int64_t AudioTokens(int64_t frames)
{
    return frames / 100 * 13 + ((frames % 100) + 7) / 8;
}

void ZeroDevice(OrtRunner& runner, Ort::Value& value)
{
    const auto status = cudaMemsetAsync(value.GetTensorMutableRawData(), 0,
                                        value.GetTensorTypeAndShapeInfo().GetElementCount() * ElementBytes(value),
                                        reinterpret_cast<cudaStream_t>(runner.compute_stream->GetHandle()));
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

Json ReadJson(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Missing model asset: " + path.string() + "; re-export the model");
    return Json::parse(file);
}

void Append(std::vector<int64_t>& dst, const std::vector<int64_t>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

Ort::Value DeviceValue(OrtRunner& runner, const std::vector<int64_t>& shape, ONNXTensorElementDataType dtype)
{
    return Ort::Value::CreateTensor(runner.DeviceAllocator(), shape.data(), shape.size(), dtype);
}

struct EncodedAudio
{
    int64_t tokens;
    Ort::Value values;
};

void CopyDevice(OrtRunner& runner, Ort::Value& dst, const Ort::Value& src, size_t count, size_t dst_offset = 0,
                size_t src_offset = 0)
{
    const auto bytes = ElementBytes(dst);
    const auto status =
        cudaMemcpyAsync(static_cast<char*>(dst.GetTensorMutableRawData()) + dst_offset * bytes,
                        static_cast<const char*>(src.GetTensorRawData()) + src_offset * bytes, count * bytes,
                        cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(runner.compute_stream->GetHandle()));
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

const din::io::Audio& NormalizeAudio(const din::io::Audio& audio, din::io::Audio& normalized)
{
    if (audio.sample_rate != kRate || audio.samples.empty())
        throw std::runtime_error("Expected nonempty mono 16 kHz audio");
    float peak = 0.f;
    for (float sample : audio.samples)
    {
        if (!std::isfinite(sample))
            throw std::runtime_error("Audio contains non-finite samples");
        peak = std::max(peak, std::abs(sample));
    }
    const auto* source = &audio;
    if (peak > 1.f)
    {
        normalized.sample_rate = kRate;
        normalized.samples.resize(audio.samples.size());
        std::transform(audio.samples.begin(), audio.samples.end(), normalized.samples.begin(),
                       [peak](float x)
                       {
                           return x / peak;
                       });
        source = &normalized;
    }
    return *source;
}

struct TextInputs
{
    Buffer<int64_t> ids, positions, logits_index;
    Ort::Value audio;
    FloatBuffer bias;
    Buffer<bool> mask;
    int64_t sequence, hidden, capacity;
    OrtRunner& runner;
    bool mask_uploaded = false;

    TextInputs(OrtRunner& runner, int64_t seq, int64_t width, int64_t keys, ONNXTensorElementDataType dtype)
        : ids(runner, {1, seq}, true)
        , positions(runner, {1, seq}, true)
        , logits_index(runner, {1}, true)
        , audio(DeviceValue(runner, {1, seq, width}, dtype))
        , bias(runner, {1, 1, seq, keys}, dtype)
        , mask(runner, {1, seq, 1}, true)
        , sequence(seq)
        , hidden(width)
        , capacity(keys)
        , runner(runner)
    {
        ZeroDevice(runner, audio);
        mask.Fill(false);
    }

    void Fill(std::span<const int64_t> tokens, int64_t start, int64_t audio_id, const EncodedAudio* embeddings,
              int64_t audio_offset = 0)
    {
        if (tokens.empty() || tokens.size() > static_cast<size_t>(sequence) || start + sequence > capacity)
            throw std::runtime_error("Input exceeds the exported context capacity");
        int64_t audio_pos = audio_offset;
        bool mask_changed = false;
        for (int64_t i = 0; i < sequence; ++i)
        {
            ids.HostData()[i] = i < static_cast<int64_t>(tokens.size()) ? tokens[i] : 0;
            positions.HostData()[i] = start + i;
            const bool is_audio = embeddings && i < static_cast<int64_t>(tokens.size()) && tokens[i] == audio_id;
            mask_changed |= mask.HostData()[i] != is_audio;
            mask.HostData()[i] = is_audio;
            if (is_audio)
            {
                if (audio_pos >= embeddings->tokens)
                    throw std::runtime_error("Too many audio placeholders");
                ++audio_pos;
            }
        }
        bias.WithHost(
            [&](auto* data)
            {
                using T = std::remove_pointer_t<decltype(data)>;
                for (int64_t i = 0; i < sequence; ++i)
                {
                    const int64_t visible = start + i + 1;
                    std::fill_n(data + i * capacity, visible, T(0.f));
                    std::fill_n(data + i * capacity + visible, capacity - visible, T(-1e4f));
                }
            });
        logits_index.HostData()[0] = tokens.size() - 1;
        logits_index.CopyAsyncToDevice();
        ids.CopyAsyncToDevice();
        positions.CopyAsyncToDevice();
        // Audio placeholders form contiguous runs; assemble embeddings directly on
        // the shared stream instead of downloading each encoder window to the CPU.
        audio_pos = audio_offset;
        for (int64_t i = 0; embeddings && i < static_cast<int64_t>(tokens.size());)
        {
            if (tokens[i] != audio_id)
            {
                ++i;
                continue;
            }
            const int64_t begin = i;
            while (i < static_cast<int64_t>(tokens.size()) && tokens[i] == audio_id)
                ++i;
            CopyDevice(runner, audio, embeddings->values, (i - begin) * hidden, begin * hidden, audio_pos * hidden);
            audio_pos += i - begin;
        }
        if (!mask_uploaded || mask_changed)
        {
            mask.CopyAsyncToDevice();
            mask_uploaded = true;
        }
        bias.CopyAsyncToDevice();
    }

    void Bind(Ort::IoBinding& binding, bool decoder = false)
    {
        if (decoder)
            binding.BindInput("logits_index", logits_index.BindingValue());
        binding.BindInput("input_ids", ids.BindingValue());
        binding.BindInput("position_ids", positions.BindingValue());
        binding.BindInput("audio_embeddings", audio);
        binding.BindInput("audio_mask", mask.BindingValue());
        binding.BindInput("attention_bias", bias.BindingValue());
    }
};
}  // namespace

struct Qwen3Pipeline::Impl
{
    struct Model
    {
        Json metadata, native;
        ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        std::unique_ptr<din::io::Tokenizer> tokenizer;
        std::unique_ptr<OrtRunner> encoder, text, prefill;
        int64_t hidden = 0, audio_id = 0, window = 0;

        struct EncoderBuffers
        {
            int64_t frames, tokens;
            FloatBuffer mel, bias;
            Buffer<int64_t> indices;
            Ort::Value output;
            Ort::IoBinding binding;

            EncoderBuffers(OrtRunner& runner, int64_t count, int64_t hidden, ONNXTensorElementDataType dtype)
                : frames(count)
                , tokens(AudioTokens(count))
                , mel(runner, {(count + 99) / 100, 128, 100}, dtype, true)
                , bias(runner, {1, 1, tokens, tokens}, dtype, true)
                , indices(runner, {tokens}, true)
                , output(DeviceValue(runner, {tokens, hidden}, dtype))
                , binding(runner.session)
            {
                // A call contains exactly one independent HF encoder window.
                bias.Fill(0.f);
                for (int64_t i = 0; i < tokens; ++i)
                    indices.HostData()[i] = i;
                bias.CopyAsyncToDevice();
                indices.CopyAsyncToDevice();
                binding.BindInput("mel_chunks", mel.BindingValue());
                binding.BindInput("valid_indices", indices.BindingValue());
                binding.BindInput("attention_bias", bias.BindingValue());
                binding.BindOutput("audio_embeddings", output);
            }
        };

        // Pad the last encoder window so all audio lengths reuse one GPU shape.
        std::unique_ptr<EncoderBuffers> full_window;

        EncodedAudio Encode(const std::vector<float>& features, int64_t frames)
        {
            const int64_t tokens = AudioTokens(frames);
            EncodedAudio result{tokens, DeviceValue(*encoder, {tokens, hidden}, dtype)};
            Ort::RunOptions options;
            options.AddConfigEntry("disable_synchronize_execution_providers", "1");
            int64_t token_offset = 0;
            for (int64_t offset = 0; offset < frames; offset += window)
            {
                din::common::nvtx_scoped_range range{"qwen3.encoder_window"};
                const auto count = std::min(window, frames - offset);
                if (!full_window)
                    full_window = std::make_unique<EncoderBuffers>(*encoder, window, hidden, dtype);
                auto& input = *full_window;
                const auto valid_tokens = AudioTokens(count);
                input.bias.WithHost(
                    [&](auto* data)
                    {
                        using T = std::remove_pointer_t<decltype(data)>;
                        for (int64_t i = 0; i < input.tokens; ++i)
                        {
                            std::fill_n(data + i * input.tokens, valid_tokens, T(0.f));
                            std::fill_n(data + i * input.tokens + valid_tokens, input.tokens - valid_tokens, T(-1e4f));
                        }
                    });
                input.bias.CopyAsyncToDevice();
                input.mel.Fill(0.f);
                input.mel.WithHost(
                    [&](auto* data)
                    {
                        using T = std::remove_pointer_t<decltype(data)>;
                        for (int64_t c = 0; c < (count + 99) / 100; ++c)
                            for (int64_t m = 0; m < 128; ++m)
                                for (int64_t f = 0; f < 100 && c * 100 + f < count; ++f)
                                    data[(c * 128 + m) * 100 + f] = T(features[m * frames + offset + c * 100 + f]);
                    });
                // Complete only the upload before reusing host staging. The next
                // window's CPU preparation can overlap this window's inference.
                input.mel.UploadAndWait();
                encoder->session.Run(options, input.binding);
                CopyDevice(*encoder, result.values, input.output, valid_tokens * hidden, token_offset * hidden);
                token_offset += valid_tokens;
            }
            return result;
        }
    };

    Qwen3Config config;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "din_asr_qwen3"};
    Ort::SyncStream stream{nullptr};
    std::unique_ptr<OrtRunner> mel;
    Model asr;
    std::unique_ptr<Model> aligner;
    int64_t capacity = 0;
    std::vector<int64_t> eos;
    std::vector<Ort::Value> cache;
    std::unique_ptr<TextInputs> step, prefill;
    Ort::Value logits{nullptr};
    std::unique_ptr<Buffer<int64_t>> next_token;
    std::unique_ptr<Ort::IoBinding> decode_binding, prefill_binding;
    Ort::RunOptions decode_options;

    explicit Impl(Qwen3Config cfg, bool alignment_only = false)
        : config(std::move(cfg))
    {
        if (config.max_new_tokens <= 0)
            throw std::runtime_error("max-new-tokens must be positive");
        config.max_chunk_seconds = detail::ResolveChunkSeconds(config.max_chunk_seconds, !config.aligner_dir.empty());
        decode_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
        din::common::RegisterTensorRTRTXProvider(env);
        stream = din::common::CreateTensorRTRTXComputeStream(env);
        if (!alignment_only)
        {
            LoadModel(asr, config.model_dir, "asr");
            eos = asr.metadata["eos_token_ids"].get<std::vector<int64_t>>();
            config.lang_id = detail::AsrLanguage(config.lang_id, asr.native["languages"]);
        }
        mel = Runner(alignment_only ? config.aligner_dir : config.model_dir, "mel", "samples:1x8000",
                     "samples:1x2880000", "samples:1x19280000");
        if (!config.aligner_dir.empty())
        {
            aligner = std::make_unique<Model>();
            LoadModel(*aligner, config.aligner_dir, "aligner");
        }
    }

    void PrepareCache()
    {
        if (!cache.empty())
            return;
        capacity = asr.metadata.at("cache_capacity");
        const auto& c = asr.metadata["text_config"];
        const std::vector<int64_t> shape{1, c["num_key_value_heads"], capacity, c["head_dim"]};
        for (int i = 0; i < 2 * c["num_hidden_layers"].get<int>(); ++i)
            cache.push_back(DeviceValue(*asr.text, shape, asr.dtype));
        step = std::make_unique<TextInputs>(*asr.text, 1, asr.hidden, capacity, asr.dtype);
        prefill = std::make_unique<TextInputs>(*asr.prefill, kPrefillBlock, asr.hidden, capacity, asr.dtype);
        logits = DeviceValue(*asr.text, {1, c["vocab_size"]}, asr.dtype);
        next_token = std::make_unique<Buffer<int64_t>>(*asr.text, std::vector<int64_t>{1}, true);
        decode_binding = std::make_unique<Ort::IoBinding>(asr.text->session);
        prefill_binding = std::make_unique<Ort::IoBinding>(asr.prefill->session);
        step->Bind(*decode_binding, true);
        prefill->Bind(*prefill_binding, true);
        BindDecode(*decode_binding);
        BindDecode(*prefill_binding);
    }

    std::unique_ptr<OrtRunner> Runner(const std::filesystem::path& dir, const std::string& name, const std::string& min,
                                      const std::string& opt, const std::string& max, const std::string& variant = "")
    {
        din::common::ModelProfile profile;
        profile.min_shapes = min;
        profile.opt_shapes = opt;
        profile.max_shapes = max;
        const auto metadata = ReadJson(dir / "metadata.json");
        const auto identity = metadata.contains("graphs") && metadata["graphs"].contains(name)
                                  ? metadata["graphs"][name].get<std::string>().substr(0, 16)
                                  : dir.filename().string();
        profile.cache_subpath = "qwen3_" + name + "_" + identity + "_fixed" + variant;
        if (name == "decoder")
            profile.cache_subpath += "_kv" + asr.metadata["cache_capacity"].dump();
        if (name == "aligner")
            profile.cache_subpath += "_bins";
        profile.enable_cuda_graph = name != "aligner";
        if (!profile.enable_cuda_graph)
        {
            // Alignment runs once per chunk with changing shapes. Keep graphs
            // for repeated encoder windows and fixed-shape AR steps.
            profile.extra_ep_options.emplace_back("enable_cuda_graph", "0");
            profile.cache_subpath += "_no_graph";
        }
        profile.embed_ep_context = false;
        // ORT names external engines by graph, so different profiles need separate directories.
        return std::make_unique<OrtRunner>(
            env, (dir / (name + ".onnx")).string(), "trt-rtx", config.ep_cache_dir.string(),
            din::common::EpContextOptions{(config.ep_context_dir / profile.cache_subpath).string(), config.progress},
            profile, &stream);
    }

    void LoadModel(Model& model, const std::filesystem::path& dir, const std::string& task)
    {
        model.metadata = ReadJson(dir / "metadata.json");
        model.native = ReadJson(dir / "native.json");
        const auto& meta = model.metadata;
        if (meta.at("format_version") != (task == "asr" ? 3 : 2) || meta.at("task") != task)
            throw std::runtime_error("Re-export Qwen3 for the unified in-place decoder");
        if (task == "asr" &&
            (meta.at("prefill_block") != kPrefillBlock || meta.at("cache_capacity").get<int64_t>() < kPrefillBlock ||
             meta.at("cache_capacity").get<int64_t>() % kPrefillBlock))
            throw std::runtime_error("Unsupported Qwen3 cache geometry");
        const auto precision = meta.at("dtype").get<std::string>();
        if (precision == "bfloat16")
            model.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        else if (precision == "float16")
            model.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        else if (precision == "float32")
            model.dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        else
            throw std::runtime_error("Unsupported Qwen3 export precision: " + precision);
        if (task == "aligner" && !meta.value("timestamp_bins", false))
            throw std::runtime_error("Re-export the aligner with --only aligner for GPU timestamp selection");
        if (meta["audio_config"]["num_mel_bins"] != 128 || meta["audio_config"]["n_window"] != 50 ||
            meta["audio_config"]["n_window_infer"] != 800)
            throw std::runtime_error("Unsupported audio geometry");
        model.window = meta["audio_config"]["n_window_infer"];
        model.hidden = meta["text_config"]["hidden_size"];
        model.audio_id = meta["audio_token_id"];
        model.tokenizer = std::make_unique<din::io::Tokenizer>((dir / "processor/tokenizer.json").string(),
                                                               din::io::TokenizerFormat::ByteBpeJson);
        auto enc_shapes = [](int chunks, int tokens)
        {
            return "mel_chunks:" + std::to_string(chunks) + "x128x100,valid_indices:" + std::to_string(tokens) +
                   ",attention_bias:1x1x" + std::to_string(tokens) + "x" + std::to_string(tokens);
        };
        model.encoder = Runner(dir, "encoder", enc_shapes(8, 104), enc_shapes(8, 104), enc_shapes(8, 104));
        const int64_t cap = task == "asr" ? meta["cache_capacity"].get<int64_t>() : 8192;
        auto text_shapes = [&](int64_t seq, int slots, int64_t keys)
        {
            const auto s = std::to_string(seq);
            auto result = "input_ids:1x" + s + ",audio_embeddings:1x" + s + "x" + std::to_string(model.hidden) +
                          ",audio_mask:1x" + s + "x1,position_ids:1x" + s + ",attention_bias:1x1x" + s + "x" +
                          std::to_string(task == "asr" ? keys : seq);
            if (task == "aligner")
                result += ",timestamp_indices:" + std::to_string(slots);
            if (task == "asr")
                result += ",logits_index:1";
            return result;
        };
        const auto min = text_shapes(task == "asr" ? 1 : 4, 2, cap);
        const auto opt = text_shapes(task == "asr" ? 1 : 128, 32, cap);
        const auto max = text_shapes(task == "asr" ? kPrefillBlock : cap, 4096, cap);
        model.text = Runner(dir, task == "asr" ? "decoder" : "aligner", min, task == "asr" ? min : opt,
                            task == "asr" ? min : max);
        if (task == "asr")
            model.prefill = Runner(dir, "decoder", max, max, max, "_prefill");
        if (!model.encoder->HasDeviceIo() || !model.text->HasDeviceIo())
            throw std::runtime_error("GPU I/O is required");
    }

    void BindDecode(Ort::IoBinding& binding)
    {
        for (size_t i = 0; i < cache.size(); ++i)
        {
            binding.BindInput(("past_" + std::to_string(i)).c_str(), cache[i]);
            binding.BindOutput(("present_" + std::to_string(i)).c_str(), cache[i]);
        }
        binding.BindOutput("logits", logits);
        binding.BindOutput("next_token", next_token->BindingValue());
    }

    std::vector<float> Features(std::span<const float> audio)
    {
        din::common::nvtx_scoped_range range{"qwen3.mel"};
        const int64_t samples = std::max<int64_t>(8000, audio.size());
        Buffer<float> input(*mel, {1, samples}, true), output(*mel, {1, 128, samples / 160}, true);
        input.Fill(0.f);
        std::copy(audio.begin(), audio.end(), input.HostData());
        input.CopyAsyncToDevice();
        Ort::IoBinding binding(mel->session);
        binding.BindInput("samples", input.BindingValue());
        binding.BindOutput("features", output.BindingValue());
        mel->session.Run(Ort::RunOptions{}, binding);
        output.CopyAsyncToHostWithNotification().Sync();
        return {output.HostData(), output.HostData() + 128 * (samples / 160)};
    }

    void Align(TranscriptionResult& result, const std::vector<float>& features, int64_t frames)
    {
        din::common::nvtx_scoped_range range{"qwen3.align"};
        if (result.text.empty())
            return;
        auto& model = *aligner;
        const auto words = detail::AlignmentUnits(result.text, result.language);
        if (words.empty())
            return;
        if (words.size() > 2048)
            throw std::runtime_error("Native alignment is limited to 2048 alignment units per chunk");
        auto audio = [&]
        {
            din::common::nvtx_scoped_range range{"qwen3.align_encoder"};
            return model.Encode(features, frames);
        }();
        std::vector<int64_t> ids = model.native["audio_start"];
        ids.insert(ids.end(), audio.tokens, model.audio_id);
        Append(ids, model.native["audio_end"].get<std::vector<int64_t>>());
        std::vector<int64_t> slots;
        const int64_t timestamp_id = model.metadata["timestamp_token_id"];
        for (const auto& word : words)
        {
            Append(ids, model.tokenizer->Encode(word, false));
            for (int i = 0; i < 2; ++i)
            {
                slots.push_back(ids.size());
                ids.push_back(timestamp_id);
            }
        }
        const auto seq = static_cast<int64_t>(ids.size());
        if (seq > 8192)
            throw std::runtime_error("Alignment exceeds the native context limit");
        TextInputs input(*model.text, seq, model.hidden, seq, model.dtype);
        input.Fill(ids, 0, model.audio_id, &audio);
        const int64_t labels = model.metadata["num_labels"];
        Buffer<int64_t> indices(*model.text, {static_cast<int64_t>(slots.size())}, true);
        auto logits = DeviceValue(*model.text, {1, static_cast<int64_t>(slots.size()), labels}, model.dtype);
        Buffer<int64_t> output(*model.text, {1, static_cast<int64_t>(slots.size())}, true);
        std::copy(slots.begin(), slots.end(), indices.HostData());
        indices.CopyAsyncToDevice();
        Ort::IoBinding binding(model.text->session);
        input.Bind(binding);
        binding.BindInput("timestamp_indices", indices.BindingValue());
        binding.BindOutput("timestamp_logits", logits);
        binding.BindOutput("timestamp_bins", output.BindingValue());
        {
            din::common::nvtx_scoped_range range{"qwen3.align_inference"};
            model.text->session.Run(Ort::RunOptions{}, binding);
        }
        {
            din::common::nvtx_scoped_range range{"qwen3.align_download"};
            output.CopyAsyncToHostWithNotification().Sync();
        }
        din::common::nvtx_scoped_range postprocess{"qwen3.align_postprocess"};
        const float timestamp_scale = model.metadata["timestamp_segment_time"];
        std::vector<int> raw;
        for (size_t i = 0; i < slots.size(); ++i)
        {
            const auto best = output.HostData()[i];
            if (best < 0 || best >= labels)
                throw std::runtime_error("Invalid timestamp bin");
            raw.push_back(static_cast<int>(best * timestamp_scale));
        }
        const auto times = detail::FixTimestamps(raw);
        for (size_t i = 0; i < words.size(); ++i)
            result.timestamps.push_back({words[i], times[2 * i] / 1000.f, times[2 * i + 1] / 1000.f});
    }

    TranscriptionResult TranscribeChunk(std::span<const float> audio)
    {
        if (audio.empty() || audio.size() > kMaxSamples)
            throw std::runtime_error("Expected nonempty mono 16 kHz audio, at most 1205 seconds per chunk");
        auto features = Features(audio);
        const int64_t frames = features.size() / 128;
        auto encoded = asr.Encode(features, frames);
        std::vector<int64_t> ids = asr.native["prefixes"][config.lang_id];
        ids.insert(ids.end(), encoded.tokens, asr.audio_id);
        if (!asr.native.contains("suffixes"))
            throw std::runtime_error("Re-export native prompt assets with --only mel for official language forcing");
        Append(ids, asr.native["suffixes"][config.lang_id].get<std::vector<int64_t>>());
        PrepareCache();
        if (ids.size() + config.max_new_tokens > static_cast<size_t>(capacity))
            throw std::runtime_error("Prompt and generation budget exceed the exported KV capacity");
        // Clear on the shared CUDA stream. Unused NaN cache values can poison attention even when masked.
        for (auto& value : cache)
            ZeroDevice(*asr.text, value);
        int64_t audio_offset = 0;
        for (size_t offset = 0; offset < ids.size(); offset += kPrefillBlock)
        {
            din::common::nvtx_scoped_range range{"qwen3.prefill"};
            const auto block =
                std::span<const int64_t>(ids).subspan(offset, std::min<size_t>(kPrefillBlock, ids.size() - offset));
            prefill->Fill(block, offset, asr.audio_id, &encoded, audio_offset);
            audio_offset += std::count(block.begin(), block.end(), asr.audio_id);
            asr.prefill->session.Run(Ort::RunOptions{}, *prefill_binding);
            prefill_binding->SynchronizeOutputs();
        }
        int64_t position = ids.size();
        TranscriptionResult result;
        for (int count = 0; count < config.max_new_tokens; ++count)
        {
            din::common::nvtx_scoped_range range{"qwen3.decode_step"};
            next_token->CopyAsyncToHostWithNotification().Sync();
            const auto token = next_token->HostData()[0];
            result.tokens.push_back(token);
            if (std::find(eos.begin(), eos.end(), token) != eos.end())
            {
                result.reached_eos = true;
                break;
            }
            if (position >= capacity || count + 1 == config.max_new_tokens)
                break;
            step->Fill(std::span(&token, 1), position++, asr.audio_id, nullptr);
            asr.text->session.Run(decode_options, *decode_binding);
        }
        auto text_tokens = result.tokens;
        if (result.reached_eos)
            text_tokens.pop_back();
        const auto parsed = detail::ParseOutput(asr.tokenizer->Decode(text_tokens, false),
                                                asr.native["languages"][config.lang_id].get<std::string>());
        result.language = parsed.first;
        result.text = parsed.second;
        if (aligner)
            Align(result, features, frames);
        return result;
    }

    std::vector<WordTimestamp> AlignAudio(const din::io::Audio& audio, const std::string& text,
                                          const std::string& language)
    {
        if (audio.samples.size() > 180 * kRate)
            throw std::invalid_argument("Standalone alignment accepts up to 180 seconds; supply audio/text segments");
        din::io::Audio normalized;
        const auto& source = NormalizeAudio(audio, normalized);
        if (config.progress)
            config.progress({din::common::ProgressStage::Transcribing, "Aligning supplied text", 0, audio.Duration()});
        const auto features = Features(source.samples);
        TranscriptionResult result;
        result.text = text;
        result.language = language;
        Align(result, features, features.size() / 128);
        if (config.progress)
            config.progress(
                {din::common::ProgressStage::Transcribing, "Alignment complete", audio.Duration(), audio.Duration()});
        return result.timestamps;
    }

    TranscriptionResult Transcribe(const din::io::Audio& audio)
    {
        din::common::nvtx_scoped_range range{"qwen3.transcribe"};
        const auto start = std::chrono::steady_clock::now();
        din::io::Audio normalized;
        const auto* source = &NormalizeAudio(audio, normalized);
        if (config.progress)
            config.progress({din::common::ProgressStage::Transcribing, "Transcribing", 0, audio.Duration()});
        TranscriptionResult result;
        result.reached_eos = true;
        std::string previous_language;
        const int64_t prompt =
            asr.native["prefixes"][config.lang_id].size() + asr.native["suffixes"][config.lang_id].size();
        const int64_t available = asr.metadata["cache_capacity"].get<int64_t>() - prompt - config.max_new_tokens;
        const int seconds = static_cast<int>(available / 13);
        if (seconds < 1)
            throw std::invalid_argument(
                "Generation budget leaves no room for audio; increase cache-capacity or reduce max-new-tokens");
        // Preserve upstream boundaries when possible; reserve the +5s quiet search margin.
        const int target = std::min(config.max_chunk_seconds, seconds - 5);
        if (source->samples.size() > static_cast<size_t>(seconds) * kRate && target < 6)
            throw std::invalid_argument("KV capacity is too small for long-form quiet-boundary splitting");
        const auto chunks = source->samples.size() <= static_cast<size_t>(seconds) * kRate && target < 6
                                ? std::vector<detail::AudioChunk>{{0, source->samples.size()}}
                                : detail::SplitAudio(source->samples, std::max(6, target));
        for (const auto chunk : chunks)
        {
            auto part = TranscribeChunk(std::span(source->samples).subspan(chunk.begin, chunk.end - chunk.begin));
            result.text += part.text;  // Upstream joins literally, without overlap or inserted separators.
            Append(result.tokens, part.tokens);
            if (!part.language.empty() && part.language != previous_language)
            {
                if (!result.language.empty())
                    result.language += ",";
                result.language += part.language;
                previous_language = part.language;
            }
            const float offset = static_cast<float>(chunk.begin) / kRate;
            for (auto word : part.timestamps)
            {
                word.start_time = std::nearbyint((word.start_time + offset) * 1000.f) / 1000.f;
                word.end_time = std::nearbyint((word.end_time + offset) * 1000.f) / 1000.f;
                result.timestamps.push_back(std::move(word));
            }
            ++result.chunks_processed;
            result.reached_eos = result.reached_eos && part.reached_eos;
            if (config.progress)
                config.progress(
                    {din::common::ProgressStage::Transcribing,
                     "Completed chunk " + std::to_string(result.chunks_processed),
                     chunk.end == source->samples.size() ? audio.Duration() : static_cast<float>(chunk.end) / kRate,
                     audio.Duration()});
        }
        result.audio_seconds = static_cast<float>(audio.samples.size()) / kRate;
        result.transcribe_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        return result;
    }
};

Qwen3Pipeline::Qwen3Pipeline(Qwen3Config config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}
Qwen3Pipeline::~Qwen3Pipeline() = default;
TranscriptionResult Qwen3Pipeline::Transcribe(const din::io::Audio& audio)
{
    return impl_->Transcribe(audio);
}
TranscriptionResult Qwen3Pipeline::TranscribeFile(const std::filesystem::path& path)
{
    if (impl_->config.progress)
        impl_->config.progress({din::common::ProgressStage::DecodingAudio, path.filename().string()});
    return Transcribe(din::io::LoadAudio(path.string(), kRate));
}
Qwen3ForcedAligner::Qwen3ForcedAligner(Qwen3Config config)
{
    if (config.aligner_dir.empty())
        config.aligner_dir = "artifacts/qwen3/aligner-onnx-bf16";
    impl_ = std::make_unique<Qwen3Pipeline::Impl>(std::move(config), true);
}
Qwen3ForcedAligner::~Qwen3ForcedAligner() = default;
std::vector<WordTimestamp> Qwen3ForcedAligner::Align(const din::io::Audio& audio, const std::string& text,
                                                     const std::string& language)
{
    return impl_->AlignAudio(audio, text, language);
}
std::vector<WordTimestamp> Qwen3ForcedAligner::AlignFile(const std::filesystem::path& path, const std::string& text,
                                                         const std::string& language)
{
    if (impl_->config.progress)
        impl_->config.progress({din::common::ProgressStage::DecodingAudio, path.filename().string()});
    return Align(din::io::LoadAudio(path.string(), kRate), text, language);
}
}  // namespace din::asr::qwen3
