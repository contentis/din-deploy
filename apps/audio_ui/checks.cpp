// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "preferences.h"
#include "workbench.h"
#include <nlohmann/json.hpp>
namespace din::studio
{
void VerifyWorkbench(const std::filesystem::path& path)
{
    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup()
        {
            std::error_code ec;
            for (const auto* suffix : {"", ".json", ".json.tmp"})
            {
                auto file = path;
                file += suffix;
                std::filesystem::remove(file, ec);
            }
            auto blocked = path;
            blocked += ".blocked";
            std::filesystem::remove(blocked / "keep", ec);
            std::filesystem::remove(blocked, ec);
        }
    } cleanup{path};
    std::string expected;
    {
        Workbench first(nullptr, false);
        first.preferences_path_ = path;
        first.model_ = 3;
        first.view_ = 1;
        first.follow_playhead_ = false;
        first.import_directory_ = path.parent_path();
        first.export_directory_ = path.parent_path() / "exports";
        for (size_t i = 0; i < first.options_.size(); ++i)
        {
            auto& option = first.options_[i];
            option.directory = "model " + std::to_string(i);
            option.aligner_directory = "aligner " + std::to_string(i);
            option.language = "en";
            option.provider = "trt-rtx";
            option.max_tokens = 1024;
            option.encoder_frames = 8192;
            option.cpu_sampling = true;
        }
        first.model_ = 1;
        first.SetTimingModel("whisper aligner");
        if (std::string(first.options_[0].aligner_directory) != "aligner 0" ||
            std::string(first.options_[1].aligner_directory) != "whisper aligner")
            throw std::runtime_error("Aligner selection changed the wrong model");
        first.model_ = 3;
        first.clips_.emplace_back();
        auto& clip = first.clips_.back();
        clip.name = "quote\"\n";
        clip.result.text = "Unicode: 日本語\n\t\\";
        clip.result.timings = {{clip.result.text, .123456789, std::nullopt}, {"end", 1, 2}};
        auto exported = path;
        exported += ".json";
        first.Export(0, exported, true);
        const auto document = nlohmann::json::parse(std::ifstream(exported));
        if (document.at("text") != clip.result.text || document.at("source") != clip.name ||
            !document.at("timings")[0].at("end").is_null() ||
            document.at("timings")[0].at("start").get<double>() != .123456789 ||
            document.at("timings")[1].at("end") != 2)
            throw std::runtime_error("Transcript JSON lost text or timing precision");
        auto read = [](const std::filesystem::path& file)
        {
            std::ifstream input(file, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(input), {});
        };
        const auto previous = read(exported);
        clip.result.text.push_back(char(0xE3));  // A byte-level token can end mid-character.
        bool rejected = false;
        try
        {
            first.Export(0, exported, true);
        }
        catch (const nlohmann::json::type_error&)
        {
            rejected = true;
        }
        if (!rejected || read(exported) != previous)
            throw std::runtime_error("Failed JSON serialization damaged the previous export");
        auto neighbour = exported;
        neighbour += ".tmp";
        {
            std::ofstream out(neighbour);
            out << "Keep neighbouring file";
        }
        clip.result.text = "Replacement transcript";
        first.Export(0, exported, false);
        if (read(exported) != clip.result.text + '\n' || read(neighbour) != "Keep neighbouring file")
            throw std::runtime_error("Text export failed or overwrote a neighbouring temporary file");
        auto blocked = path;
        blocked += ".blocked";
        std::filesystem::create_directory(blocked);
        {
            std::ofstream out(blocked / "keep");
            out << "Keep destination";
        }
        rejected = false;
        try
        {
            first.Export(0, blocked, false);
        }
        catch (const std::exception&)
        {
            rejected = true;
        }
        if (!rejected || read(blocked / "keep") != "Keep destination")
            throw std::runtime_error("Failed replacement damaged the destination");
        expected = first.PreferencesText();
        // Destruction must flush edits even before the debounce interval expires.
    }
    Workbench second(nullptr, false);
    second.preferences_path_ = path;
    second.LoadPreferences();
    if (second.PreferencesText() != expected)
        throw std::runtime_error("Workbench selections were not restored");
}
int SelfTest()
{
    try
    {
        VerifyTranscriptAlignment();
        Audio source;
        source.sample_rate = 16000;
        source.samples.resize(1600, .25f);
        const auto path =
            std::filesystem::temp_directory_path() /
            ("din-audio-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".wav");
        struct Cleanup
        {
            std::filesystem::path path;
            ~Cleanup()
            {
                std::error_code ec;
                std::filesystem::remove(path, ec);
            }
        } cleanup{path};
        SaveWave(path, source);
        VerifyFileDialogs(path);
        VerifyPreferences(path.string() + ".prefs");
        VerifyWorkbench(path.string() + ".session");
        Workbench background(nullptr, false);
        auto wait_for_background = [&]
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            do
            {
                background.Update();  // No ImGui context or Draw: also runs while minimized.
                if (background.CanClose())
                    return;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            } while (std::chrono::steady_clock::now() < deadline);
            throw std::runtime_error("Background completion requires rendering");
        };
        background.Import(path);
        background.Import(path.string() + ".missing.wav");
        wait_for_background();
        background.DialogResult(2, path.string() + ".missing-model");
        background.TranscribeSelected();
        wait_for_background();
        Worker worker;
        worker.Import(7, path);
        worker.Import(8, path.string() + ".missing");
        worker.Transcribe(9, std::make_shared<Audio>(source), Settings{.directory = path.string() + ".missing-model"});
        int received = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (received < 3 && std::chrono::steady_clock::now() < deadline)
        {
            for (const auto& c : worker.Poll())
            {
                if (c.index == 7 &&
                    (c.error.size() || !c.audio || c.audio->samples.size() != 1600 || c.peaks[0] != .25f))
                    throw std::runtime_error("Audio decode/peak contract failed");
                if (c.index == 8 && c.error.empty())
                    throw std::runtime_error("Import error was lost");
                if (c.index == 9 && c.error.empty())
                    throw std::runtime_error("Inference error was lost");
                ++received;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (received != 3 || worker.Progress())
            throw std::runtime_error("Worker did not finish queued jobs and clear progress");
        Inference inference;
        bool rejected = false;
        try
        {
            inference.Run(Settings{.model = 99}, source);
        }
        catch (const std::exception&)
        {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Invalid model was accepted");
        std::cout << "PASS: WAV round trip, dialog starting folders, peaks, queued jobs, import/inference errors, "
                     "progress cleanup, invalid model\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

int AlignmentCheck(Settings settings, const Audio& audio)
{
    try
    {
        if (settings.model != 1 || settings.aligner_directory.empty())
            throw std::runtime_error("Alignment check requires Whisper and an aligner folder");
        Inference inference;
        auto native_settings = settings;
        native_settings.aligner_directory.clear();
        const auto native = inference.Run(native_settings, audio);
        if (native.text.empty() || native.timing_kind != "segment" || native.timings.empty())
            throw std::runtime_error("Whisper native segment timing was discarded");
        const auto aligned = inference.Run(settings, audio);
        if (aligned.text != native.text || aligned.timing_kind != "word" || aligned.timings.empty() ||
            !aligned.warning.empty())
            throw std::runtime_error("Alignment changed transcription or failed: " + aligned.warning);
        for (const auto& word : aligned.timings)
            if (!word.end || word.start < 0 || *word.end < word.start || *word.end > audio.Duration() + .01)
                throw std::runtime_error("Alignment has invalid full-recording timestamps");
        settings.aligner_directory += "/intentionally-missing-alignment-check";
        const auto failed = inference.Run(settings, audio);
        if (failed.text != native.text || failed.timing_kind != native.timing_kind || failed.warning.empty() ||
            failed.timings.size() != native.timings.size())
            throw std::runtime_error("Alignment loading failure discarded native transcription");
        for (size_t i = 0; i < native.timings.size(); ++i)
            if (failed.timings[i].text != native.timings[i].text ||
                failed.timings[i].start != native.timings[i].start || failed.timings[i].end != native.timings[i].end)
                throw std::runtime_error("Alignment failure changed native segment timing");
        std::cout << "PASS: " << native.timings.size() << " native segments, " << aligned.timings.size()
                  << " aligned words; identical transcript and failure recovery over " << audio.Duration() << " s\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
}  // namespace din::studio
