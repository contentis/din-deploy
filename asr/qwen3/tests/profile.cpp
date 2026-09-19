// SPDX-License-Identifier: Apache-2.0
#include <cuda_profiler_api.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "qwen3.h"
#include <nlohmann/json.hpp>

// Internal profiling driver: warm the same shapes/addresses before capturing.
int main(int argc, char** argv)
{
    try
    {
        if (argc < 2 || argc > 7)
            throw std::runtime_error(
                "Usage: din_asr_qwen3_profile AUDIO [SECONDS [TOKENS [ALIGNER_DIR [RESULT_JSON [MODEL_DIR]]]]]");
        din::asr::qwen3::Qwen3Config config;
        if (argc > 6)
            config.model_dir = argv[6];
        if (argc > 4)
            config.aligner_dir = argv[4];
        config.max_new_tokens = argc > 3 ? std::stoi(argv[3]) : 256;
        const int seconds = argc > 2 ? std::stoi(argv[2]) : 90;
        if (seconds <= 0)
            throw std::runtime_error("SECONDS must be positive");
        din::asr::qwen3::Qwen3Pipeline pipeline(config);
        auto audio = din::io::LoadAudio(argv[1], 16000);
        if (audio.samples.size() > static_cast<size_t>(seconds) * 16000)
            audio.samples.resize(static_cast<size_t>(seconds) * 16000);
        const auto warm = pipeline.Transcribe(audio);
        if (cudaProfilerStart() != cudaSuccess)
            throw std::runtime_error("Cannot start CUDA profiler capture");
        const auto measured = pipeline.Transcribe(audio);
        if (warm.tokens != measured.tokens)
        {
            const auto count = std::min(warm.tokens.size(), measured.tokens.size());
            size_t first = 0;
            while (first < count && warm.tokens[first] == measured.tokens[first])
                ++first;
            std::cerr << "First mismatch: " << first << ", token counts: " << warm.tokens.size() << "/"
                      << measured.tokens.size() << '\n';
            throw std::runtime_error("Warm/measured token mismatch");
        }
        if (warm.timestamps.size() != measured.timestamps.size())
            throw std::runtime_error("Warm/measured timestamp count mismatch");
        for (size_t i = 0; i < warm.timestamps.size(); ++i)
        {
            const auto& a = warm.timestamps[i];
            const auto& b = measured.timestamps[i];
            if (a.text != b.text || a.start_time != b.start_time || a.end_time != b.end_time)
                throw std::runtime_error("Warm/measured timestamp mismatch");
        }
        std::cout << "Warm/measured tokens match. Tokens: " << measured.tokens.size()
                  << ", words: " << measured.timestamps.size() << ", chunks: " << measured.chunks_processed
                  << ", EOS: " << measured.reached_eos << ", audio seconds: " << measured.audio_seconds
                  << ", seconds: " << measured.transcribe_seconds << std::endl;
        if (argc > 5)
        {
            nlohmann::json result{{"tokens", measured.tokens},
                                  {"transcription", measured.text},
                                  {"reached_eos", measured.reached_eos},
                                  {"chunks_processed", measured.chunks_processed},
                                  {"audio_seconds", measured.audio_seconds},
                                  {"transcribe_seconds", measured.transcribe_seconds}};
            result["timestamps"] = nlohmann::json::array();
            for (const auto& word : measured.timestamps)
                result["timestamps"].push_back(
                    {{"text", word.text}, {"start_time", word.start_time}, {"end_time", word.end_time}});
            std::ofstream file(argv[5]);
            file << result.dump(2) << '\n';
            if (!file)
                throw std::runtime_error("Cannot write profiling result");
        }
        if (cudaProfilerStop() != cudaSuccess)
            throw std::runtime_error("Cannot stop CUDA profiler capture");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
