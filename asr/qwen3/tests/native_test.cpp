// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "internal/chunks.h"
#include "internal/text.h"
#include "qwen3.h"
#include "tokenizer.h"
#include <nlohmann/json.hpp>

void Require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
    using namespace din::asr::qwen3;
    try
    {
        Require(detail::SplitAudio({}).empty(), "Empty split mismatch");
        std::vector<float> split_audio(61 * 16000, 1.f);
        std::fill(split_audio.begin() + 24 * 16000, split_audio.begin() + 24 * 16000 + 1600, 0.f);
        const auto chunks = detail::SplitAudio(split_audio, 25);
        Require(chunks.size() == 3 && chunks[0].end == 24 * 16000 && chunks[1].end == 44 * 16000 &&
                    chunks[2].end == split_audio.size(),
                "Quiet boundary or first-tie mismatch");
        for (size_t i = 0; i < chunks.size(); ++i)
            Require(chunks[i].begin == (i ? chunks[i - 1].end : 0) && chunks[i].end - chunks[i].begin <= 480000,
                    "Chunk gap, overlap or capacity violation");
        Require(detail::SplitAudio(std::vector<float>(400000), 25).size() == 1, "Exact target must stay whole");
        Require(detail::SplitAudio(std::vector<float>(181 * 16000)).size() == 1,
                "Official ASR target must preserve a three-minute recording");
        Require(detail::SplitAudio(std::vector<float>(181 * 16000), 180).size() == 2,
                "Official alignment target must split at three minutes");
        Require(detail::EnglishWords(" Well, don't stop! ") == std::vector<std::string>{"Well", "don't", "stop"},
                "English normalization mismatch");
        Require(detail::ParseOutput(" language eNGLISH\n<asr_text> hello ") ==
                    std::pair<std::string, std::string>{"English", "hello"},
                "Official metadata parsing mismatch");
        Require(detail::ParseOutput("language None<asr_text>noise").second == "noise", "None parsing mismatch");
        Require(detail::ParseOutput(std::string(21, 'a'), "English").second == "a", "Repetition repair mismatch");
        std::string repeated;
        for (int i = 0; i < 20; ++i)
            repeated += "ab";
        Require(detail::ParseOutput(repeated).second == "ab", "Pattern repair mismatch");
        Require(detail::EnglishWords("he\xe2\x80\x94said don\xe2\x80\x99t") ==
                    std::vector<std::string>{"hesaid", "dont"},
                "HF typographic punctuation mismatch");
        Require(detail::FixTimestamps({0, 240, 80, 160, 320}) == std::vector<int>{0, 0, 80, 160, 320},
                "Short timestamp outlier repair mismatch");
        Require(detail::FixTimestamps({0, 800, 720, 640, 80, 160, 240, 320}) ==
                    std::vector<int>{0, 20, 40, 60, 80, 160, 240, 320},
                "Timestamp interpolation mismatch");
        if (argc == 1)
            return 0;
        if ((argc == 4 || argc == 5) && std::string(argv[1]) == "--chunks")
        {
            const auto audio = din::io::LoadAudio(argv[2], 16000);
            std::ofstream pcm(argv[3], std::ios::binary);
            pcm.write(reinterpret_cast<const char*>(audio.samples.data()), audio.samples.size() * sizeof(float));
            Require(static_cast<bool>(pcm), "Cannot write reference PCM");
            nlohmann::json boundaries = nlohmann::json::array();
            for (const auto chunk : detail::SplitAudio(audio.samples, argc == 5 ? std::stoul(argv[4]) : 1200))
                boundaries.push_back({chunk.begin, chunk.end});
            std::cout << boundaries.dump() << '\n';
            return 0;
        }
        if (argc != 4 && argc != 6)
            throw std::runtime_error("Usage: native_test [MODEL AUDIO ASR_REPORT [ALIGNER ALIGN_REPORT]]");
        Qwen3Config config;
        config.model_dir = argv[1];
        din::io::Tokenizer tokenizer((config.model_dir / "processor/tokenizer.json").string(),
                                     din::io::TokenizerFormat::ByteBpeJson);
        std::ifstream tokenizer_file(std::filesystem::path(__FILE__).parent_path() / "tokenizer.reference.json");
        for (const auto& item : nlohmann::json::parse(tokenizer_file))
        {
            const auto ids = item["ids"].get<std::vector<int64_t>>();
            const auto text = item["text"].get<std::string>();
            Require(tokenizer.Decode(ids, false) == text, "HF byte decoding mismatch");
            if (item["ascii_word"].get<bool>())
                Require(tokenizer.Encode(text, false) == ids, "HF English word encoding mismatch");
        }
        if (argc == 6)
            config.aligner_dir = argv[4];
        std::ifstream report_file(argv[3]);
        const auto report = nlohmann::json::parse(report_file);
        const auto expected = report["cases"][0]["reference_tokens"].get<std::vector<int64_t>>();
        Qwen3Pipeline pipeline(config);
        const auto audio = din::io::LoadAudio(argv[2], 16000);
        auto first = pipeline.Transcribe(audio);
        Require(first.reached_eos && first.tokens == expected, "Native/HF token mismatch");
        if (argc == 6)
        {
            std::ifstream alignment_file(argv[5]);
            const auto spans = nlohmann::json::parse(alignment_file)["reference_spans"];
            Require(first.timestamps.size() == spans.size(), "Timestamp count mismatch");
            for (size_t i = 0; i < first.timestamps.size(); ++i)
                Require(first.timestamps[i].text == spans[i]["text"].get<std::string>() &&
                            first.timestamps[i].start_time == spans[i]["start_time"].get<float>() &&
                            first.timestamps[i].end_time == spans[i]["end_time"].get<float>(),
                        "HF word span mismatch");
        }
        auto silence = pipeline.Transcribe({std::vector<float>(32000, 0.f), 16000});
        Require(silence.reached_eos && silence.text.empty() && silence.timestamps.empty(), "Silence regression");
        auto repeat = pipeline.Transcribe(audio);
        Require(repeat.tokens == first.tokens && repeat.text == first.text, "Cache reset/reuse regression");
        const auto long_silence = pipeline.Transcribe({std::vector<float>(480001, 0.f), 16000});
        Require(long_silence.reached_eos && long_silence.text.empty() && long_silence.chunks_processed == 1,
                "Long silence or chunk cache reset regression");
        std::cout << "Native tokens, silence, repeated calls and input bounds passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
