// SPDX-License-Identifier: Apache-2.0
#include "qwen3.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <span>
#include <stdexcept>

#include "internal/chunks.h"
#include "internal/text.h"
#include "nvtx_helper.h"
#include "ort_session.h"
#include "tokenizer.h"
#include <nlohmann/json.hpp>

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

// Same shared TensorBuffer staging and stream ordering as the other ASR samples.
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

        // Bound retained storage to a full window and the current tail, keeping
        // addresses stable across full windows and repeated same-shape requests.
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

    explicit Impl(Qwen3Config cfg)
        : config(std::move(cfg))
    {
        if (config.max_new_tokens <= 0)
            throw std::runtime_error("max-new-tokens must be positive");
        decode_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
        din::common::RegisterTensorRTRTXProvider(env);
        stream = din::common::CreateTensorRTRTXComputeStream(env);
        LoadModel(asr, config.model_dir, "asr");
        if (!asr.native["prefixes"].contains(config.lang_id))
            throw std::runtime_error("Unknown language hint");
        mel = Runner(config.model_dir, "mel", "samples:1x8000", "samples:1x2880000", "samples:1x19280000");
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
        if (result.language != "English")
            throw std::runtime_error("Native forced alignment currently supports English only");
        auto& model = *aligner;
        const auto words = detail::EnglishWords(result.text);
        if (words.empty())
            return;
        if (words.size() > 2048)
            throw std::runtime_error("Native alignment is limited to 2048 words per chunk");
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

    TranscriptionResult Transcribe(const din::io::Audio& audio)
    {
        din::common::nvtx_scoped_range range{"qwen3.transcribe"};
        if (audio.sample_rate != kRate || audio.samples.empty())
            throw std::runtime_error("Expected nonempty mono 16 kHz audio");
        if (std::any_of(audio.samples.begin(), audio.samples.end(),
                        [](float x)
                        {
                            return !std::isfinite(x);
                        }))
            throw std::runtime_error("Audio contains non-finite samples");
        const auto start = std::chrono::steady_clock::now();
        const float peak = std::abs(*std::max_element(audio.samples.begin(), audio.samples.end(),
                                                      [](float a, float b)
                                                      {
                                                          return std::abs(a) < std::abs(b);
                                                      }));
        din::io::Audio normalized;
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
        TranscriptionResult result;
        result.reached_eos = true;
        std::string previous_language;
        for (const auto chunk : detail::SplitAudio(source->samples, aligner ? 180 : 1200))
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
}  // namespace din::asr::qwen3
