// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "audio.h"
#include "forced_aligner.h"
#include "progress.h"

namespace din::asr::qwen3
{
struct Qwen3Config
{
    std::string provider = "trt-rtx";
    std::filesystem::path model_dir = "artifacts/qwen3/onnx-bf16";
    std::string aligner_provider;       // Empty uses the ASR provider.
    std::filesystem::path aligner_dir;  // Empty disables forced alignment.
    std::filesystem::path ep_cache_dir = "artifacts/qwen3/rt_cache";
    std::filesystem::path ep_context_dir = "artifacts/qwen3/ep_context";
    std::string lang_id = "auto";
    int max_new_tokens = 1024;
    int max_chunk_seconds = 0;  // Auto: 1200 s ASR / 180 s aligned, limited by KV capacity; boundaries add up to 5 s.
    din::common::ProgressCallback progress;
    // Optional callers can prepare the aligner on a separate thread after ASR construction.
    bool defer_alignment = false;
    din::common::ProgressCallback alignment_progress;
};

struct TranscriptionResult
{
    std::string text;
    std::string language;
    std::vector<int64_t> tokens;
    std::vector<WordTimestamp> timestamps;
    bool reached_eos = false;
    size_t chunks_processed = 0;
    float audio_seconds = 0;
    float transcribe_seconds = 0;
};

// One synchronous stream per instance. Reuse the instance across files.
class Qwen3Pipeline
{
public:
    explicit Qwen3Pipeline(Qwen3Config config);
    ~Qwen3Pipeline();
    Qwen3Pipeline(const Qwen3Pipeline&) = delete;
    Qwen3Pipeline& operator=(const Qwen3Pipeline&) = delete;
    void PrepareAlignment();
    // When preparing concurrently, this callback must join preparation before alignment accesses it.
    TranscriptionResult Transcribe(const din::io::Audio& audio, const std::function<void()>& wait_for_alignment = {});
    TranscriptionResult TranscribeFile(const std::filesystem::path& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace din::asr::qwen3
