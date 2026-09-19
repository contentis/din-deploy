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

#include "nvtx_helper.h"
#include "ort_session.h"
#include "tokenizer.h"
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

std::vector<std::string> EnglishWords(const std::string& text)
{
    std::vector<std::string> words;
    std::string word;
    std::string cleaned = text;
    // HF drops typographic punctuation, including curly apostrophes (only ASCII ' is kept).
    for (const auto* punctuation : {"\xe2\x80\x93", "\xe2\x80\x94", "\xe2\x80\x98", "\xe2\x80\x99", "\xe2\x80\x9c",
                                    "\xe2\x80\x9d", "\xe2\x80\xa6"})
    {
        size_t pos = 0;
        while ((pos = cleaned.find(punctuation, pos)) != std::string::npos)
            cleaned.erase(pos, 3);
    }
    for (unsigned char c : cleaned)
    {
        if (c >= 128)
            throw std::runtime_error("Native alignment currently requires ASCII English text");
        if (std::isspace(c))
        {
            if (!word.empty())
                words.push_back(std::move(word));
            word.clear();
        }
        else if (std::isalnum(c) || c == '\'')
            word.push_back(static_cast<char>(c));
    }
    if (!word.empty())
        words.push_back(std::move(word));
    return words;
}

std::vector<std::string> AlignmentUnits(const std::string& text, const std::string& language)
{
    if (language == "English" || language == "en")
        return EnglishWords(text);
    if (language != "Chinese" && language != "zh" && language != "Cantonese" && language != "yue")
        throw std::invalid_argument("Native alignment supports English, Chinese and Cantonese");
    std::vector<std::string> units;
    std::string word;
    const auto flush = [&]
    {
        if (!word.empty())
            units.push_back(std::move(word));
        word.clear();
    };
    for (size_t i = 0; i < text.size();)
    {
        const auto begin = i;
        const auto first = static_cast<unsigned char>(text[i++]);
        const int bytes = first < 0x80                     ? 1
                          : first >= 0xc2 && first <= 0xdf ? 2
                          : first >= 0xe0 && first <= 0xef ? 3
                          : first >= 0xf0 && first <= 0xf4 ? 4
                                                           : 0;
        if (!bytes || begin + bytes > text.size())
            throw std::invalid_argument("Invalid UTF-8 transcript");
        uint32_t code = first & (bytes == 1 ? 0x7f : bytes == 2 ? 0x1f : bytes == 3 ? 0x0f : 0x07);
        for (int j = 1; j < bytes; ++j)
        {
            const auto next = static_cast<unsigned char>(text[i++]);
            if ((next & 0xc0) != 0x80)
                throw std::invalid_argument("Invalid UTF-8 transcript");
            code = (code << 6) | (next & 0x3f);
        }
        if ((bytes == 2 && code < 0x80) || (bytes == 3 && code < 0x800) || (bytes == 4 && code < 0x10000) ||
            code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
            throw std::invalid_argument("Invalid UTF-8 transcript");
        const bool cjk = (code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf) ||
                         (code >= 0x20000 && code <= 0x2a6df) || (code >= 0x2a700 && code <= 0x2b73f) ||
                         (code >= 0x2b740 && code <= 0x2b81f) || (code >= 0x2b820 && code <= 0x2ceaf) ||
                         (code >= 0xf900 && code <= 0xfaff) || (code >= 0x2f800 && code <= 0x2fa1f);
        if (cjk)
        {
            flush();
            units.push_back(text.substr(begin, bytes));
        }
        else if (code < 128)
        {
            if (std::isspace(static_cast<unsigned char>(code)))
                flush();
            else if (std::isalnum(static_cast<unsigned char>(code)) || code == '\'')
                word.push_back(static_cast<char>(code));
        }
        else if (code == 0x3000 || code == 0xa0 || (code >= 0x2000 && code <= 0x200a) || code == 0x2028 ||
                 code == 0x2029)
            flush();
        else if ((code >= 0x2010 && code <= 0x2027) || (code >= 0x3001 && code <= 0x301f) ||
                 (code >= 0xff01 && code <= 0xff0f) || (code >= 0xff1a && code <= 0xff20) ||
                 (code >= 0xff3b && code <= 0xff40) || (code >= 0xff5b && code <= 0xff65))
            continue;
        else
            throw std::invalid_argument("Unsupported non-CJK character in alignment transcript");
    }
    flush();
    return units;
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
constexpr int kRate = 16000;
constexpr int kMaxSamples = 1205 * kRate;
constexpr int kPrefillBlock = 512;

int64_t AudioTokens(int64_t frames)
{
    return frames / 100 * 13 + ((frames % 100) + 7) / 8;
}

void ZeroDevice(OrtRunner& runner, Ort::Value& value)
{
    const auto status = cudaMemsetAsync(value.GetTensorMutableData<BF16>(), 0,
                                        value.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(BF16),
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

Ort::Value DeviceValue(OrtRunner& runner, const std::vector<int64_t>& shape)
{
    return Ort::Value::CreateTensor<BF16>(runner.DeviceAllocator(), shape.data(), shape.size());
}

struct EncodedAudio
{
    int64_t tokens;
    Ort::Value values;
};

void CopyDevice(OrtRunner& runner, BF16* dst, const BF16* src, size_t count)
{
    const auto status = cudaMemcpyAsync(dst, src, count * sizeof(BF16), cudaMemcpyDeviceToDevice,
                                        reinterpret_cast<cudaStream_t>(runner.compute_stream->GetHandle()));
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

const din::io::Audio& NormalizeAudio(const din::io::Audio& audio, din::io::Audio& normalized)
{
    if (audio.sample_rate != kRate || audio.samples.empty())
        throw std::runtime_error("Expected nonempty mono 16 kHz audio");
    if (std::any_of(audio.samples.begin(), audio.samples.end(),
                    [](float x)
                    {
                        return !std::isfinite(x);
                    }))
        throw std::runtime_error("Audio contains non-finite samples");
    const float peak = std::abs(*std::max_element(audio.samples.begin(), audio.samples.end(),
                                                  [](float a, float b)
                                                  {
                                                      return std::abs(a) < std::abs(b);
                                                  }));
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
    Buffer<int64_t> ids, positions;
    Ort::Value audio;
    Buffer<BF16> bias;
    Buffer<bool> mask;
    int64_t sequence, hidden, capacity;
    OrtRunner& runner;
    bool mask_uploaded = false;

    TextInputs(OrtRunner& runner, int64_t seq, int64_t width, int64_t keys)
        : ids(runner, {1, seq}, true)
        , positions(runner, {1, seq}, true)
        , audio(DeviceValue(runner, {1, seq, width}))
        , bias(runner, {1, 1, seq, keys}, true)
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
        if (tokens.size() != static_cast<size_t>(sequence) || start + sequence > capacity)
            throw std::runtime_error("Input exceeds the exported context capacity");
        int64_t audio_pos = audio_offset;
        bool mask_changed = false;
        for (int64_t i = 0; i < sequence; ++i)
        {
            ids.HostData()[i] = tokens[i];
            positions.HostData()[i] = start + i;
            const bool is_audio = embeddings && tokens[i] == audio_id;
            mask_changed |= mask.HostData()[i] != is_audio;
            mask.HostData()[i] = is_audio;
            if (is_audio)
            {
                if (audio_pos >= embeddings->tokens)
                    throw std::runtime_error("Too many audio placeholders");
                ++audio_pos;
            }
            const int64_t visible = start + i + 1;
            std::fill_n(bias.HostData() + i * capacity, visible, BF16(0.f));
            std::fill_n(bias.HostData() + i * capacity + visible, capacity - visible, BF16(-1e4f));
        }
        ids.CopyAsyncToDevice();
        positions.CopyAsyncToDevice();
        // Audio placeholders form contiguous runs; assemble embeddings directly on
        // the shared stream instead of downloading each encoder window to the CPU.
        audio_pos = audio_offset;
        for (int64_t i = 0; embeddings && i < sequence;)
        {
            if (tokens[i] != audio_id)
            {
                ++i;
                continue;
            }
            const int64_t begin = i;
            while (i < sequence && tokens[i] == audio_id)
                ++i;
            CopyDevice(runner, audio.GetTensorMutableData<BF16>() + begin * hidden,
                       embeddings->values.GetTensorData<BF16>() + audio_pos * hidden, (i - begin) * hidden);
            audio_pos += i - begin;
        }
        if (!mask_uploaded || mask_changed)
        {
            mask.CopyAsyncToDevice();
            mask_uploaded = true;
        }
        bias.CopyAsyncToDevice();
    }

    void Bind(Ort::IoBinding& binding)
    {
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
        std::unique_ptr<din::io::Tokenizer> tokenizer;
        std::unique_ptr<OrtRunner> encoder, text, decode_alt;
        int64_t hidden = 0, audio_id = 0;

        struct EncoderBuffers
        {
            int64_t frames, tokens;
            Buffer<BF16> mel, bias;
            Buffer<int64_t> indices;
            Ort::Value output;
            Ort::IoBinding binding;

            EncoderBuffers(OrtRunner& runner, int64_t count, int64_t hidden)
                : frames(count)
                , tokens(AudioTokens(count))
                , mel(runner, {(count + 99) / 100, 128, 100}, true, true)
                , bias(runner, {1, 1, tokens, tokens}, true)
                , indices(runner, {tokens}, true)
                , output(DeviceValue(runner, {tokens, hidden}))
                , binding(runner.session)
            {
                // A call contains exactly one independent HF encoder window.
                bias.Fill(BF16(0.f));
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

        // Stable addresses for repeated full windows and the current tail.
        std::unique_ptr<EncoderBuffers> full_window, tail_window;

        EncodedAudio Encode(const std::vector<float>& features, int64_t frames)
        {
            const int64_t window = metadata["audio_config"]["n_window_infer"];
            const int64_t tokens = AudioTokens(frames);
            EncodedAudio result{tokens, DeviceValue(*encoder, {tokens, hidden})};
            Ort::RunOptions options;
            options.AddConfigEntry("disable_synchronize_execution_providers", "1");
            int64_t token_offset = 0;
            for (int64_t offset = 0; offset < frames; offset += window)
            {
                din::common::nvtx_scoped_range range{"qwen3.encoder_window"};
                const auto count = std::min(window, frames - offset);
                auto& buffers = count == window ? full_window : tail_window;
                if (!buffers || buffers->frames != count)
                    buffers = std::make_unique<EncoderBuffers>(*encoder, count, hidden);
                auto& input = *buffers;
                input.mel.Fill(BF16(0.f));
                for (int64_t c = 0; c < (count + 99) / 100; ++c)
                    for (int64_t m = 0; m < 128; ++m)
                        for (int64_t f = 0; f < 100 && c * 100 + f < count; ++f)
                            input.mel.HostData()[(c * 128 + m) * 100 + f] =
                                BF16(features[m * frames + offset + c * 100 + f]);
                // Complete only the upload before reusing host staging. The next
                // window's CPU preparation can overlap this window's inference.
                input.mel.CopyAsyncToDeviceWithNotification().Sync();
                encoder->session.Run(options, input.binding);
                CopyDevice(*encoder, result.values.GetTensorMutableData<BF16>() + token_offset * hidden,
                           input.output.GetTensorData<BF16>(), input.tokens * hidden);
                token_offset += input.tokens;
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
    std::array<std::vector<Ort::Value>, 2> cache;
    std::unique_ptr<TextInputs> step;
    // Retain one full block and the current tail per cache bank, with stable addresses.
    std::array<std::unique_ptr<TextInputs>, 2> prefill_full, prefill_tail;
    Ort::Value logits{nullptr};
    std::unique_ptr<Buffer<int64_t>> next_token;
    std::array<std::unique_ptr<Ort::IoBinding>, 2> decode_bindings;
    std::unique_ptr<OrtRunner> fast_decode, spare_decode;
    int64_t fast_capacity = 0, spare_capacity = 0;
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
            if (!asr.native["prefixes"].contains(config.lang_id))
                throw std::runtime_error("Unknown language hint");
        }
        mel = Runner(alignment_only ? config.aligner_dir : config.model_dir, "mel", "samples:1x8000",
                     "samples:1x2880000", "samples:1x19280000");
        if (!config.aligner_dir.empty())
        {
            aligner = std::make_unique<Model>();
            LoadModel(*aligner, config.aligner_dir, "aligner");
        }
    }

    void PrepareCache(int64_t required)
    {
        din::common::nvtx_scoped_range range{"qwen3.prepare_cache"};
        const int64_t maximum = asr.metadata.at("cache_capacity");
        int64_t requested = maximum;
        if (asr.metadata.value("dynamic_cache_capacity", false))
        {
            requested = 4;
            while (requested < required && requested < maximum)
                requested *= 2;
            requested = std::min(requested, maximum);
        }
        if (requested == capacity)
            return;
        for (auto& binding : decode_bindings)
            binding.reset();
        step.reset();
        for (auto& input : prefill_full)
            input.reset();
        for (auto& input : prefill_tail)
            input.reset();
        for (auto& buffers : cache)
            buffers.clear();
        capacity = requested;
        const auto decode_capacities = asr.metadata.value("decode_capacities", std::vector<int64_t>{});
        if (std::find(decode_capacities.begin(), decode_capacities.end(), capacity) != decode_capacities.end())
        {
            // Retain at most two engines for the usual full-chunk/tail buckets.
            // Switching chunks should not reload weights and JIT-specialize again.
            if (spare_capacity == capacity)
            {
                std::swap(fast_decode, spare_decode);
                std::swap(fast_capacity, spare_capacity);
            }
            else if (fast_capacity != capacity)
            {
                spare_decode = std::move(fast_decode);
                spare_capacity = fast_capacity;
                fast_decode = Runner(config.model_dir, "decode_" + std::to_string(capacity), "", "", "");
                fast_capacity = capacity;
            }
        }
        else if (fast_decode)
        {
            spare_decode = std::move(fast_decode);
            spare_capacity = fast_capacity;
            fast_capacity = 0;
        }
        const auto& c = asr.metadata["text_config"];
        const int layers = c["num_hidden_layers"];
        const std::vector<int64_t> shape{1, c["num_key_value_heads"], capacity, c["head_dim"]};
        for (auto& buffers : cache)
            for (int i = 0; i < 2 * layers; ++i)
                buffers.push_back(DeviceValue(*asr.text, shape));
        step = std::make_unique<TextInputs>(*asr.text, 1, asr.hidden, capacity);
        logits = DeviceValue(*asr.text, {1, c["vocab_size"]});
        next_token = std::make_unique<Buffer<int64_t>>(*asr.text, std::vector<int64_t>{1}, true);
        for (int bank = 0; bank < (fast_decode ? 1 : 2); ++bank)
        {
            auto& runner = fast_decode ? fast_decode : (bank ? asr.decode_alt : asr.text);
            decode_bindings[bank] = std::make_unique<Ort::IoBinding>(runner->session);
            step->Bind(*decode_bindings[bank]);
            BindDecode(*decode_bindings[bank], bank, fast_decode != nullptr);
        }
    }

    std::unique_ptr<OrtRunner> Runner(const std::filesystem::path& dir, const std::string& name, const std::string& min,
                                      const std::string& opt, const std::string& max)
    {
        din::common::ModelProfile profile;
        profile.min_shapes = min;
        profile.opt_shapes = opt;
        profile.max_shapes = max;
        profile.cache_subpath = dir.filename().string() + "_qwen3_profiled_" + name;
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
        return std::make_unique<OrtRunner>(
            env, (dir / (name + ".onnx")).string(), "trt-rtx", config.ep_cache_dir.string(),
            din::common::EpContextOptions{config.ep_context_dir.string(), config.progress}, profile, &stream);
    }

    void LoadModel(Model& model, const std::filesystem::path& dir, const std::string& task)
    {
        model.metadata = ReadJson(dir / "metadata.json");
        model.native = ReadJson(dir / "native.json");
        const auto& meta = model.metadata;
        if (meta.at("format_version") != 2 || meta.at("dtype") != "bfloat16" || meta.at("task") != task)
            throw std::runtime_error("Native Qwen3 requires format-2 BF16 " + task + " exports");
        if (task == "aligner" && !meta.value("timestamp_bins", false))
            throw std::runtime_error("Re-export the aligner with --only aligner for GPU timestamp selection");
        if (meta["audio_config"]["num_mel_bins"] != 128 || meta["audio_config"]["n_window"] != 50)
            throw std::runtime_error("Unsupported audio geometry");
        model.hidden = meta["text_config"]["hidden_size"];
        model.audio_id = meta["audio_token_id"];
        model.tokenizer = std::make_unique<din::io::Tokenizer>((dir / "processor/tokenizer.json").string(),
                                                               din::io::TokenizerFormat::ByteBpeJson);
        auto enc_shapes = [](int chunks, int tokens)
        {
            return "mel_chunks:" + std::to_string(chunks) + "x128x100,valid_indices:" + std::to_string(tokens) +
                   ",attention_bias:1x1x" + std::to_string(tokens) + "x" + std::to_string(tokens);
        };
        model.encoder = Runner(dir, "encoder", enc_shapes(1, 1), enc_shapes(8, 104), enc_shapes(8, 104));
        const int64_t cap = task == "asr" ? meta["cache_capacity"].get<int64_t>() : 8192;
        auto text_shapes = [&](int64_t seq, int slots, int64_t keys)
        {
            const auto s = std::to_string(seq);
            auto result = "input_ids:1x" + s + ",audio_embeddings:1x" + s + "x" + std::to_string(model.hidden) +
                          ",audio_mask:1x" + s + "x1,position_ids:1x" + s + ",attention_bias:1x1x" + s + "x" +
                          std::to_string(task == "asr" ? keys : seq);
            if (task == "aligner")
                result += ",timestamp_indices:" + std::to_string(slots);
            if (task == "asr" && meta.value("dynamic_cache_capacity", false))
            {
                const auto& c = meta["text_config"];
                for (int i = 0; i < 2 * c["num_hidden_layers"].get<int>(); ++i)
                    result += ",past_" + std::to_string(i) + ":1x" +
                              std::to_string(c["num_key_value_heads"].get<int>()) + "x" + std::to_string(keys) + "x" +
                              std::to_string(c["head_dim"].get<int>());
            }
            return result;
        };
        const auto min = text_shapes(task == "asr" ? 1 : 4, 2, meta.value("dynamic_cache_capacity", false) ? 4 : cap);
        const auto opt = text_shapes(std::min<int64_t>(128, cap), 32,
                                     meta.value("dynamic_cache_capacity", false) ? std::min<int64_t>(2048, cap) : cap);
        const auto max = text_shapes(task == "asr" ? std::min<int64_t>(kPrefillBlock, cap) : cap, 4096, cap);
        model.text = Runner(dir, task == "asr" ? "decoder" : "aligner", min, opt, max);
        // One context per bank keeps CUDA graph bindings stable during decode.
        if (task == "asr")
            model.decode_alt = Runner(dir, "decoder", min, opt, max);
        if (!model.encoder->HasDeviceIo() || !model.text->HasDeviceIo())
            throw std::runtime_error("GPU I/O is required");
    }

    void BindDecode(Ort::IoBinding& binding, int bank, bool inplace = false)
    {
        for (size_t i = 0; i < cache[bank].size(); ++i)
        {
            binding.BindInput(("past_" + std::to_string(i)).c_str(), cache[bank][i]);
            binding.BindOutput(("present_" + std::to_string(i)).c_str(), cache[inplace ? bank : 1 - bank][i]);
        }
        binding.BindOutput("logits", logits);
        binding.BindOutput("next_token", next_token->BindingValue());
    }

    void Decode(int bank)
    {
        auto& runner = fast_decode ? fast_decode : (bank ? asr.decode_alt : asr.text);
        runner->session.Run(decode_options, *decode_bindings[bank]);
    }

    std::vector<float> Features(const din::io::Audio& audio)
    {
        din::common::nvtx_scoped_range range{"qwen3.mel"};
        const int64_t samples = std::max<int64_t>(8000, audio.samples.size());
        Buffer<float> input(*mel, {1, samples}, true), output(*mel, {1, 128, samples / 160}, true);
        input.Fill(0.f);
        std::copy(audio.samples.begin(), audio.samples.end(), input.HostData());
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
        TextInputs input(*model.text, seq, model.hidden, seq);
        input.Fill(ids, 0, model.audio_id, &audio);
        const int64_t labels = model.metadata["num_labels"];
        Buffer<int64_t> indices(*model.text, {static_cast<int64_t>(slots.size())}, true);
        auto logits = DeviceValue(*model.text, {1, static_cast<int64_t>(slots.size()), labels});
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
        std::vector<int> raw;
        for (size_t i = 0; i < slots.size(); ++i)
        {
            const auto best = output.HostData()[i];
            if (best < 0 || best >= labels)
                throw std::runtime_error("Invalid timestamp bin");
            raw.push_back(static_cast<int>(best * model.metadata["timestamp_segment_time"].get<float>()));
        }
        const auto times = detail::FixTimestamps(raw);
        for (size_t i = 0; i < words.size(); ++i)
            result.timestamps.push_back({words[i], times[2 * i] / 1000.f, times[2 * i + 1] / 1000.f});
    }

    TranscriptionResult TranscribeChunk(const din::io::Audio& audio)
    {
        if (audio.sample_rate != kRate || audio.samples.empty() || audio.samples.size() > kMaxSamples)
            throw std::runtime_error("Expected nonempty mono 16 kHz audio, at most 1205 seconds per chunk");
        auto features = Features(audio);
        const int64_t frames = features.size() / 128;
        auto encoded = asr.Encode(features, frames);
        std::vector<int64_t> ids = asr.native["prefixes"][config.lang_id];
        ids.insert(ids.end(), encoded.tokens, asr.audio_id);
        if (!asr.native.contains("suffixes"))
            throw std::runtime_error("Re-export native prompt assets with --only mel for official language forcing");
        Append(ids, asr.native["suffixes"][config.lang_id].get<std::vector<int64_t>>());
        PrepareCache(ids.size() + config.max_new_tokens);
        if (ids.size() >= static_cast<size_t>(capacity))
            throw std::runtime_error("Prompt fills the exported KV capacity; re-export with --cache-capacity 32768");
        // Clear on the shared CUDA stream. Unused NaN cache values can poison attention even when masked.
        for (auto& value : cache[0])
            ZeroDevice(*asr.text, value);
        int bank = 0;
        int64_t audio_offset = 0;
        for (size_t offset = 0; offset < ids.size(); offset += kPrefillBlock)
        {
            din::common::nvtx_scoped_range range{"qwen3.prefill"};
            const auto block =
                std::span<const int64_t>(ids).subspan(offset, std::min<size_t>(kPrefillBlock, ids.size() - offset));
            auto& runner = bank ? asr.decode_alt : asr.text;
            auto& prefill = (block.size() == kPrefillBlock ? prefill_full : prefill_tail)[bank];
            if (!prefill || prefill->sequence != static_cast<int64_t>(block.size()))
                prefill = std::make_unique<TextInputs>(*runner, block.size(), asr.hidden, capacity);
            prefill->Fill(block, offset, asr.audio_id, &encoded, audio_offset);
            audio_offset += std::count(block.begin(), block.end(), asr.audio_id);
            Ort::IoBinding binding(runner->session);
            prefill->Bind(binding);
            BindDecode(binding, bank);
            runner->session.Run(Ort::RunOptions{}, binding);
            binding.SynchronizeOutputs();
            bank = 1 - bank;
        }
        if (fast_decode && bank != 0)
        {
            // One handoff copy, on the shared stream. Decode then updates this bank
            // in place with stable graph addresses; prefill remains non-aliasing.
            for (size_t i = 0; i < cache[0].size(); ++i)
                CopyDevice(*asr.text, cache[0][i].GetTensorMutableData<BF16>(), cache[bank][i].GetTensorData<BF16>(),
                           cache[0][i].GetTensorTypeAndShapeInfo().GetElementCount());
            bank = 0;
        }
        int64_t position = ids.size();
        TranscriptionResult result;
        const auto eos = asr.metadata["eos_token_ids"].get<std::vector<int64_t>>();
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
            Decode(bank);
            if (!fast_decode)
                bank = 1 - bank;
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
        const auto features = Features(NormalizeAudio(audio, normalized));
        TranscriptionResult result;
        result.text = text;
        result.language = language;
        Align(result, features, features.size() / 128);
        return result.timestamps;
    }

    TranscriptionResult Transcribe(const din::io::Audio& audio)
    {
        din::common::nvtx_scoped_range range{"qwen3.transcribe"};
        const auto start = std::chrono::steady_clock::now();
        din::io::Audio normalized;
        const auto* source = &NormalizeAudio(audio, normalized);
        TranscriptionResult result;
        result.reached_eos = true;
        std::string previous_language;
        for (const auto chunk : detail::SplitAudio(source->samples, config.max_chunk_seconds))
        {
            din::io::Audio input{{source->samples.begin() + chunk.begin, source->samples.begin() + chunk.end}, kRate};
            auto part = TranscribeChunk(input);
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
                config.progress({din::common::ProgressStage::Transcribing,
                                 "Completed chunk " + std::to_string(result.chunks_processed),
                                 double(chunk.end) / kRate, audio.Duration()});
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
    return Align(din::io::LoadAudio(path.string(), kRate), text, language);
}
}  // namespace din::asr::qwen3
