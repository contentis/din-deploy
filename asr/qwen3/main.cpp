// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

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
        parser.add_description("Offline Qwen3 ASR and forced alignment.");
        parser.add_argument("audiofile");
        parser.add_argument("--model-dir").default_value(config.model_dir.string());
        parser.add_argument("--aligner-dir").default_value(std::string{}).help("Exported forced aligner directory");
        parser.add_argument("--transcript")
            .default_value(std::string{})
            .help("Align supplied UTF-8 text file without ASR");
        parser.add_argument("--lang-id").default_value(config.lang_id);
        parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string());
        parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string());
        parser.add_argument("--max-new-tokens").default_value(config.max_new_tokens).scan<'i', int>();
        parser.add_argument("--max-chunk-seconds")
            .default_value(config.max_chunk_seconds)
            .scan<'i', int>()
            .help(
                "Chunk target: 0 = auto (1200 s ASR / 180 s aligned), limited by KV capacity; boundaries may add 5 s");
        parser.parse_args(argc, argv);
        config.model_dir = parser.get<std::string>("--model-dir");
        config.aligner_dir = parser.get<std::string>("--aligner-dir");
        config.lang_id = parser.get<std::string>("--lang-id");
        config.ep_cache_dir = parser.get<std::string>("--ep-cache");
        config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
        config.max_new_tokens = parser.get<int>("--max-new-tokens");
        config.max_chunk_seconds = parser.get<int>("--max-chunk-seconds");
        const auto transcript = parser.get<std::string>("--transcript");
        TranscriptionResult result;
        if (transcript.empty())
        {
            Qwen3Pipeline pipeline(std::move(config));
            result = pipeline.TranscribeFile(parser.get<std::string>("audiofile"));
        }
        else
        {
            std::ifstream file(transcript, std::ios::binary);
            if (!file)
                throw std::runtime_error("Cannot read transcript: " + transcript);
            result.text = {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
            if (result.text.starts_with("\xef\xbb\xbf"))
                result.text.erase(0, 3);
            const auto audio = din::io::LoadAudio(parser.get<std::string>("audiofile"), 16000);
            result.language = config.lang_id == "auto" ? "English" : config.lang_id;
            Qwen3ForcedAligner aligner(std::move(config));
            const auto start = std::chrono::steady_clock::now();
            result.timestamps = aligner.Align(audio, result.text, result.language);
            result.transcribe_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
            result.audio_seconds = audio.Duration();
            result.reached_eos = true;
            result.chunks_processed = 1;
        }
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
