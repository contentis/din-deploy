// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "audio.h"
#include "file_dialog.h"
#include "paths.h"
#include "progress.h"
#include "transcript.h"

namespace din::studio
{
using Audio = din::io::Audio;
struct Settings
{
    int model = 0;
    std::string directory, provider = "cpu";
    std::string language = "auto", aligner_directory;
    int max_tokens = 1024, encoder_frames = 65536;
    bool cpu_sampling = false;
    bool operator==(const Settings&) const = default;
};
inline constexpr const char* Models[] = {"Qwen3", "Whisper", "Parakeet TDT", "Nemotron"};
class Inference
{
public:
    Inference();
    ~Inference();
    Result Run(const Settings&, const Audio&, din::common::ProgressCallback progress = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
enum class ClipState
{
    Ready,
    Importing,
    Queued,
    Complete,
    Error
};
struct Clip
{
    std::string name;
    std::filesystem::path source;
    std::shared_ptr<const Audio> audio;
    std::vector<float> peaks;
    Result result;
    ClipState state = ClipState::Ready;
    std::string error;
    bool Pending() const
    {
        return state == ClipState::Importing || state == ClipState::Queued;
    }
    std::string Status() const
    {
        switch (state)
        {
        case ClipState::Importing:
            return "Importing";
        case ClipState::Queued:
            return "Queued";
        case ClipState::Complete:
            return result.warning.empty() ? "Complete" : "Complete with warning";
        case ClipState::Error:
            return "Error: " + error;
        default:
            return "Ready";
        }
    }
};
struct Completion
{
    size_t index = 0;
    bool importing = false;
    std::shared_ptr<const Audio> audio;
    std::vector<float> peaks;
    Result result;
    std::string error;
};
struct JobProgress
{
    size_t index = 0;
    din::common::InferenceProgress event{din::common::ProgressStage::DecodingAudio, ""};
    std::chrono::steady_clock::time_point stage_started{}, inference_started{};
    double measured_seconds = 0;
};
class Worker
{
public:
    Worker();
    ~Worker();
    void Import(size_t index, std::filesystem::path path);
    void Transcribe(size_t index, std::shared_ptr<const Audio> audio, Settings settings);
    std::vector<Completion> Poll();
    std::optional<JobProgress> Progress();
    std::atomic<bool> busy{false};

private:
    struct Job
    {
        size_t index;
        std::filesystem::path path;
        std::shared_ptr<const Audio> audio;
        Settings settings;
    };
    void Submit(Job job);
    void Loop();
    void Report(size_t index, const din::common::InferenceProgress&);
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Job> jobs_;
    std::vector<Completion> completed_;
    std::optional<JobProgress> progress_;
    bool stop_ = false;
    Inference inference_;
    std::thread thread_;
};
std::shared_ptr<Audio> Decode(const std::filesystem::path&);
std::vector<float> Peaks(const Audio&);
void SaveWave(const std::filesystem::path&, const Audio&);
class AudioDevice
{
public:
    AudioDevice();
    ~AudioDevice();
    void Play(std::shared_ptr<const Audio>, double seconds);
    void Pause();
    void Seek(double seconds);
    double Position() const;
    bool Playing() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class Workbench
{
public:
    explicit Workbench(void* window, bool persist = true);
    ~Workbench();
    void Update();
    void Draw();
    void Import(const std::filesystem::path&);
    void DialogResult(int action, const std::string& path);
    bool CanClose() const;
    bool HasTranscript() const;
    void TranscribeSelected();
    void SetTimingModel(const std::string& path);
    void ShowModelSettings(int model);
    void SelectModel(int model);

private:
    void Library();
    friend void VerifyWorkbench(const std::filesystem::path&);
    void Player();
    void Transcript();
    void DrawProgress(size_t index);
    void ModelSettings();
    void Timeline(float height);
    void Queue(size_t);
    void Seek(double);
    void Export(size_t index, const std::filesystem::path&, bool json);
    void OpenDialog(int action);
    std::filesystem::path ExpectedAligner() const;
    void LoadPreferences();
    std::string PreferencesText() const;
    void SavePreferences(bool immediately = false);
    void* window_;
    FileDialogs dialogs_;
    std::filesystem::path import_directory_ = "assets", export_directory_;
    std::filesystem::path preferences_path_;
    std::string saved_preferences_, pending_preferences_;
    std::chrono::steady_clock::time_point preferences_changed_{};
    Worker worker_;
    AudioDevice device_;
    std::vector<Clip> clips_;
    std::array<Settings, 4> options_;
    bool open_model_settings_ = false;
    int selected_ = -1, model_ = 0, view_ = 0;
    float timeline_zoom_ = 1, timeline_start_ = 0;
    std::vector<float> timing_offsets_;
    float timing_font_size_ = 0;
    bool follow_playhead_ = true, audio_was_playing_ = false;
    double position_ = 0;
    std::string error_;
};
int SelfTest();
int AlignmentCheck(Settings, const Audio&);
void VerifyWorkbench(const std::filesystem::path&);
}  // namespace din::studio
