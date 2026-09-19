// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "audio.h"
#include "progress.h"

namespace din::asr::qwen3
{
struct Qwen3Config
{
    std::filesystem::path model_dir = "artifacts/qwen3/onnx-bf16";
    std::filesystem::path aligner_dir;  // Empty disables forced alignment.
    std::filesystem::path ep_cache_dir = "artifacts/qwen3/rt_cache";
    std::filesystem::path ep_context_dir = "artifacts/qwen3/ep_context";
    std::string lang_id = "auto";
    int max_new_tokens = 512;  // Official Transformers wrapper default; increase for long speech.
    din::common::ProgressCallback progress;
};

struct WordTimestamp
{
    std::string text;
    float start_time = 0;
    float end_time = 0;
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
    TranscriptionResult Transcribe(const din::io::Audio& audio);
    TranscriptionResult TranscribeFile(const std::filesystem::path& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace din::asr::qwen3
