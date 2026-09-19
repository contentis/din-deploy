// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "qwen3.h"
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

int main(int argc, char** argv)
{
    using namespace din::asr::qwen3;
    try
    {
        Qwen3Config config;
        argparse::ArgumentParser parser("din_asr_qwen3_cli");
        parser.add_description("Offline, single-stream Qwen3 ASR with optional English forced alignment.");
        parser.add_argument("audiofile");
        parser.add_argument("--model-dir").default_value(config.model_dir.string());
        parser.add_argument("--aligner-dir")
            .default_value(std::string{})
            .help("Enable word timestamps with this exported aligner");
        parser.add_argument("--lang-id").default_value(config.lang_id);
        parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string());
        parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string());
        parser.add_argument("--max-new-tokens").default_value(config.max_new_tokens).scan<'i', int>();
        parser.parse_args(argc, argv);
        config.model_dir = parser.get<std::string>("--model-dir");
        config.aligner_dir = parser.get<std::string>("--aligner-dir");
        config.lang_id = parser.get<std::string>("--lang-id");
        config.ep_cache_dir = parser.get<std::string>("--ep-cache");
        config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
        config.max_new_tokens = parser.get<int>("--max-new-tokens");
        Qwen3Pipeline pipeline(std::move(config));
        const auto result = pipeline.TranscribeFile(parser.get<std::string>("audiofile"));
        nlohmann::json output{
            {"transcription", result.text},          {"language", result.language},
            {"reached_eos", result.reached_eos},     {"tokens", result.tokens},
            {"audio_seconds", result.audio_seconds}, {"transcribe_seconds", result.transcribe_seconds}};
        output["timestamps"] = nlohmann::json::array();
        output["chunks_processed"] = result.chunks_processed;
        for (const auto& word : result.timestamps)
            output["timestamps"].push_back(
                {{"text", word.text}, {"start_time", word.start_time}, {"end_time", word.end_time}});
        std::cout << output.dump(2) << '\n';
        return result.reached_eos ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Qwen3: " << error.what() << '\n';
        return 1;
    }
}
