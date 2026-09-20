// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "preferences.h"
#include "preparation.h"
#include "workbench.h"
#include <nlohmann/json.hpp>
namespace din::studio
{
void VerifyPreparation()
{
    std::promise<void> started, release;
    auto entered = started.get_future();
    auto gate = release.get_future();
    std::atomic<bool> second_finished = false;
    {
        Preparation preparation(
            [&]
            {
                started.set_value();
                gate.wait();
                throw std::runtime_error("injected alignment load failure");
            },
            [&]
            {
                second_finished = true;
            });
        const bool background_started = entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        const bool stayed_async =
            preparation.alignment.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
        const bool ordered = !second_finished;
        release.set_value();
        bool failed = false;
        try
        {
            preparation.alignment.get();
        }
        catch (const std::runtime_error&)
        {
            failed = true;
        }
        if (!background_started || !stayed_async || !ordered || !failed)
            throw std::runtime_error("Background preparation blocked, reordered tasks or lost a failure");
        // Leave the second future unconsumed, as on early ASR failure.
    }
    if (!second_finished)
        throw std::runtime_error("Preparation did not join on job exit");
}
void Workbench::ShowSpeakerDemo()
{
    Clip clip;
    clip.name = "Speaker demo (synthetic)";
    clip.audio = std::make_shared<Audio>(Audio{std::vector<float>(6 * 16000, 0), 16000});
    clip.peaks = Peaks(*clip.audio);
    clip.state = ClipState::Complete;
    clip.result.model = "Speaker UI preview";
    clip.result.text = "Welcome back. What should we cover today? Let's start with the audio tools. Sounds good.";
    clip.result.timing_kind = "segment";
    clip.result.timings = {{"Welcome back. What should we cover today?", 0, 2, 0},
                           {"Let's start with the audio tools.", 2, 4, 1},
                           {"Sounds good.", 4, 6, 0}};
    clip.result.speakers = {{0, "Alex"}, {1, "Sam"}};
    clip.result.speaker_activity = {{"", 0, 2.2, 0}, {"", 2, 4.1, 1}, {"", 4, 6, 0}};
    BuildSpeakerTurns(clip.result);
    clips_.push_back(std::move(clip));
    selected_ = static_cast<int>(clips_.size() - 1);
    view_ = 2;
    diarizer_.enabled = true;
}
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
        first.timeline_height_ = 320;
        first.import_directory_ = path.parent_path();
        first.export_directory_ = path.parent_path() / "exports";
        for (size_t i = 0; i < first.options_.size(); ++i)
        {
            auto& option = first.options_[i];
            option.directory = "model " + std::to_string(i);
            option.language = "en";
            option.provider = "trt-rtx";
            option.max_tokens = 1024;
            option.encoder_frames = 8192;
            option.cpu_sampling = true;
        }
        first.model_ = 1;
        first.SetTimingModel("shared aligner");
        first.SetDiarizationModel("shared diarizer");
        first.diarizer_.provider = "cpu";
        const auto submitted = first.JobSettings();
        for (int model = 0; model < 4; ++model)
        {
            first.SelectModel(model);
            if (first.JobSettings().aligner_directory != "shared aligner" ||
                first.JobSettings().diarizer_directory != "shared diarizer" ||
                first.JobSettings().diarizer_provider != "cpu")
                throw std::runtime_error("Changing transcription models lost the processing step");
        }
        first.diarizer_.enabled = false;
        if (!first.JobSettings().diarizer_directory.empty() || submitted.diarizer_directory != "shared diarizer")
            throw std::runtime_error("Diarization switch changed a submitted job");
        first.aligner_.enabled = false;
        if (!first.JobSettings().aligner_directory.empty() || first.aligner_.directory != "shared aligner" ||
            submitted.aligner_directory != "shared aligner")
            throw std::runtime_error("Disabling alignment lost its folder or changed a submitted job");
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
        first.ShowSpeakerDemo();
        // A paused/end-of-playback update must not undo a transcript seek.
        first.audio_was_playing_ = true;
        first.Seek(2);
        first.Update();
        if (first.position_ != 2 || first.device_.Playing())
            throw std::runtime_error("Paused transcript seek was lost or started playback");
        first.Seek(100);
        if (first.position_ != 6)
            throw std::runtime_error("Seek exceeded the recording duration");
        auto& demo = first.clips_.back().result;
        const auto original = demo.text;
        demo.speakers[0] = "Renamed \"host\"";
        if (demo.text != original || SpeakerText(demo).find("Renamed \"host\":") == std::string::npos ||
            demo.speaker_turns.size() != 3 || demo.speaker_activity[0].end.value() <= demo.speaker_activity[1].start)
            throw std::runtime_error("Speaker naming lost transcript, turns or overlaps");
        first.Export(first.clips_.size() - 1, exported, true);
        const auto named = nlohmann::json::parse(std::ifstream(exported));
        if (named["speakers"][0]["name"] != demo.speakers[0] || named["timings"][0]["speaker"] != 0 ||
            named["text"] != original || named["speaker_activity"].size() != 3)
            throw std::runtime_error("Speaker JSON lost names, IDs, overlaps or original text");
        first.Export(first.clips_.size() - 1, exported, false);
        if (read(exported) != SpeakerText(demo) + '\n')
            throw std::runtime_error("Speaker text export ignored renamed speakers");
        first.diarizer_.enabled = false;
        expected = first.PreferencesText();
        // Destruction must flush edits even before the debounce interval expires.
    }
    Workbench second(nullptr, false);
    second.preferences_path_ = path;
    second.LoadPreferences();
    if (second.PreferencesText() != expected)
        throw std::runtime_error("Workbench selections were not restored");
    if (second.aligner_.enabled || second.aligner_.directory.empty() || !second.JobSettings().aligner_directory.empty())
        throw std::runtime_error("Disabled alignment was re-enabled on restart");
    if (second.diarizer_.enabled || second.diarizer_.directory.empty() || second.diarizer_.provider != "cpu")
        throw std::runtime_error("Diarization preferences were not restored");
    second.aligner_.enabled = true;
    second.DialogResult(6, "replacement aligner");
    if (second.JobSettings().aligner_directory != "replacement aligner")
        throw std::runtime_error("Aligner browse selection was lost");
    second.aligner_.directory.clear();
    bool rejected = false;
    try
    {
        second.JobSettings();
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Enabled alignment without a model was silently skipped");
    // Existing per-ASR preferences migrate to an independent processing step.
    WriteFileAtomically(path, EncodePreferences({{"version", "1"},
                                                 {"selected_model", "1"},
                                                 {"model.0.aligner", "qwen aligner"},
                                                 {"model.1.aligner", "legacy whisper aligner"}}));
    Workbench migrated(nullptr, false);
    migrated.preferences_path_ = path;
    migrated.LoadPreferences();
    if (!migrated.aligner_.enabled || migrated.JobSettings().aligner_directory != "legacy whisper aligner")
        throw std::runtime_error("Legacy aligner selection was not migrated");
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
        // Exercise conversion through the same loader used by CLI models.
        for (const int rate : {8000, 44100, 48000})
        {
            Audio input;
            input.sample_rate = rate;
            input.samples.resize(rate / 10, .25f);
            SaveWave(path, input);
            const auto converted = din::io::LoadAudio(path, 16000);
            if (converted.sample_rate != 16000 || std::abs(converted.Duration() - .1) > .001)
                throw std::runtime_error("Resampling changed the recording duration");
            for (size_t i = 32; i + 32 < converted.samples.size(); ++i)
                if (std::abs(converted.samples[i] - .25f) > .001f)
                    throw std::runtime_error("Resampling changed the signal level");
        }
        for (const int rate : {0, -16000})
        {
            bool rejected = false;
            try
            {
                din::io::LoadAudio(path, rate);
            }
            catch (const std::invalid_argument&)
            {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Invalid audio sample rate was accepted");
        }
        std::ofstream(path, std::ios::binary).close();
        bool empty_rejected = false;
        try
        {
            din::io::LoadAudio(path, 16000);
        }
        catch (const std::runtime_error&)
        {
            empty_rejected = true;
        }
        if (!empty_rejected)
            throw std::runtime_error("Empty recording was accepted");
        SaveWave(path, source);
        auto unicode_path = path;
        unicode_path += std::filesystem::path(u8"-日本語.wav");
        Cleanup unicode_cleanup{unicode_path};
        SaveWave(unicode_path, source);
        if (din::io::LoadAudio(unicode_path, 16000).samples != source.samples)
            throw std::runtime_error("Unicode audio path or lossless float decoding failed");
        VerifyFileDialogs(path);
        VerifyPreferences(path.string() + ".prefs");
        VerifyWorkbench(path.string() + ".session");
        VerifyPreparation();
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
        std::cout << "PASS: WAV round trip, resampling, invalid/empty audio, dialog starting folders, peaks, queued "
                     "jobs, import/inference errors, "
                     "progress cleanup, background preparation, invalid model\n";
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
        const auto caller = std::this_thread::get_id();
        std::atomic<bool> prepared_in_background = false;
        const auto aligned = inference.Run(settings, audio, {},
                                           [&](int, const std::string&)
                                           {
                                               if (std::this_thread::get_id() != caller)
                                                   prepared_in_background = true;
                                           });
        if (!prepared_in_background || !std::isfinite(aligned.seconds) || aligned.seconds <= 0 ||
            aligned.setup_seconds < 0)
            throw std::runtime_error("Optional preparation did not run in background or broke time accounting");
        if (aligned.text != native.text || aligned.timing_kind != "word" || aligned.timings.empty() ||
            !aligned.warning.empty())
            throw std::runtime_error("Alignment changed transcription or failed: " + aligned.warning);
        for (const auto& word : aligned.timings)
            if (!word.end || word.start < 0 || *word.end < word.start || *word.end > audio.Duration() + .01)
                throw std::runtime_error("Alignment has invalid full-recording timestamps");
        const auto cached = inference.Run(settings, audio);
        if (cached.text != aligned.text || cached.timings.size() != aligned.timings.size() || !cached.warning.empty())
            throw std::runtime_error("Cached optional models failed on a repeat run");
        settings.aligner_directory += "/intentionally-missing-alignment-check";
        const auto failed = inference.Run(settings, audio);
        if (failed.text != native.text || failed.timing_kind != native.timing_kind || failed.warning.empty() ||
            failed.timings.size() != native.timings.size())
            throw std::runtime_error("Alignment loading failure discarded native transcription");
        for (size_t i = 0; i < native.timings.size(); ++i)
            if (failed.timings[i].text != native.timings[i].text ||
                failed.timings[i].start != native.timings[i].start || failed.timings[i].end != native.timings[i].end)
                throw std::runtime_error("Alignment failure changed native segment timing");
        settings.aligner_directory.clear();
        settings.diarizer_directory = "intentionally-missing-diarization-check";
        const auto no_speakers = inference.Run(settings, audio);
        if (no_speakers.text != native.text || no_speakers.warning.empty() || !no_speakers.speakers.empty())
            throw std::runtime_error("Diarization preparation failure discarded transcription");
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
