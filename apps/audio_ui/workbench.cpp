// SPDX-License-Identifier: Apache-2.0
#include "workbench.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "imgui.h"
#include "imgui_support.h"
#include "misc/cpp/imgui_stdlib.h"
#include "preferences.h"
#include <nlohmann/json.hpp>
#include <SDL3/SDL.h>

namespace din::studio
{
namespace
{
const char* StageName(din::common::ProgressStage stage)
{
    switch (stage)
    {
    case din::common::ProgressStage::DecodingAudio:
        return "Decoding audio";
    case din::common::ProgressStage::LoadingModel:
        return "Loading model";
    case din::common::ProgressStage::CompilingModel:
        return "Compiling model";
    case din::common::ProgressStage::Transcribing:
        return "Transcribing";
    case din::common::ProgressStage::Aligning:
        return "Aligning words";
    }
    return "Processing";
}
}  // namespace
Workbench::Workbench(void* window, bool persist)
    : window_(window)
{
    const char* defaults[] = {"artifacts/qwen3/onnx-bf16", "artifacts/whisper/onnx", "artifacts/parakeet/onnx",
                              "artifacts/nemotron/onnx"};
    for (size_t i = 0; i < options_.size(); ++i)
        options_[i].directory = defaults[i];
    if (persist)
        LoadPreferences();
}
Workbench::~Workbench()
{
    SavePreferences(true);
}
bool Workbench::HasTranscript() const
{
    return std::any_of(clips_.begin(), clips_.end(),
                       [](const Clip& clip)
                       {
                           return !clip.result.model.empty();
                       });
}
bool Workbench::CanClose() const
{
    return !dialogs_.Busy() && !worker_.busy.load() &&
           std::none_of(clips_.begin(), clips_.end(),
                        [](const Clip& c)
                        {
                            return c.Pending();
                        });
}
void Workbench::TranscribeSelected()
{
    if (selected_ >= 0)
        Queue(static_cast<size_t>(selected_));
}
void Workbench::Import(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c)
                   {
                       return char(std::tolower(c));
                   });
    if (ext != ".wav" && ext != ".mp3" && ext != ".flac")
    {
        error_ = "Choose WAV, MP3 or FLAC audio.";
        return;
    }
    Clip c;
    c.name = Utf8(path.filename());
    c.source = std::filesystem::absolute(path);
    import_directory_ = c.source.parent_path();
    c.state = ClipState::Importing;

    clips_.push_back(std::move(c));
    worker_.Import(clips_.size() - 1, path);
    if (selected_ < 0)
        selected_ = 0;
}
void Workbench::DialogResult(int action, const std::string& path)
{
    try
    {
        if (action < 0)
            error_ = path;
        else if (action == 1)
            Import(Utf8Path(path));
        else if (action % 8 == 2)
        {
            auto& option = options_.at(static_cast<size_t>(action / 8));
            option.directory = path;
        }
        else if (action % 8 == 6)
        {
            auto& option = options_.at(static_cast<size_t>(action / 8));
            option.aligner_directory = path;
        }
        else
        {
            const auto index = static_cast<size_t>(action / 8);
            action %= 8;
            Export(index, Utf8Path(path), action == 4);
        }
    }
    catch (const std::exception& e)
    {
        error_ = e.what();
    }
}
void Workbench::Queue(size_t index)
{
    auto& clip = clips_[index];
    if (!clip.audio || clip.Pending())
        return;
    auto settings = options_[model_];
    settings.model = model_;
    if (settings.language.empty())
        settings.language = "auto";
    if (model_ == 0)
        settings.provider = "trt-rtx";

    clip.state = ClipState::Queued;
    worker_.Transcribe(index, clip.audio, std::move(settings));
}
void Workbench::Seek(double seconds)
{
    position_ = seconds;
    try
    {
        if (device_.Playing() && selected_ >= 0)
            device_.Seek(seconds);
    }
    catch (const std::exception& e)
    {
        error_ = e.what();
    }
}
void Workbench::Library()
{
    const auto progress = worker_.Progress();
    if (ImGui::Button("+ Import audio", ImVec2(-1, 0)))
        OpenDialog(1);
    ImGui::TextDisabled("WAV, MP3, FLAC");
    ImGui::Spacing();
    for (size_t i = 0; i < clips_.size(); ++i)
    {
        ImGui::PushID(static_cast<int>(i));
        const auto row = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        const float line = ImGui::GetTextLineHeight();
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, ImGui::GetStyle().ItemSpacing.y));
        if (ImGui::Selectable("##clip", selected_ == static_cast<int>(i), 0, ImVec2(0, line * 2 + 18)))
        {
            try
            {
                device_.Pause();
            }
            catch (const std::exception& e)
            {
                error_ = e.what();
            }
            selected_ = static_cast<int>(i);
            timing_offsets_.clear();
            position_ = 0;
            audio_was_playing_ = false;
            timeline_start_ = 0;
            timeline_zoom_ = 1;
        }
        ImGui::PopStyleVar();
        auto* draw = ImGui::GetWindowDrawList();
        draw->PushClipRect(row, ImVec2(row.x + width, row.y + line * 2 + 18), true);
        draw->AddText(ImVec2(row.x + 8, row.y + 5), ImGui::GetColorU32(ImGuiCol_Text), clips_[i].name.c_str());
        std::ostringstream detail;
        if (clips_[i].audio)
            detail << std::fixed << std::setprecision(1) << clips_[i].audio->Duration() << " s  /  ";
        detail << (progress && progress->index == i ? StageName(progress->event.stage) : clips_[i].Status().c_str());
        draw->AddText(ImVec2(row.x + 8, row.y + line + 8), ImGui::GetColorU32(ImGuiCol_TextDisabled),
                      detail.str().c_str());
        draw->PopClipRect();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s\n%s", clips_[i].name.c_str(), detail.str().c_str());
        ImGui::PopID();
    }
}
void Workbench::Player()
{
    if (selected_ < 0 || !clips_[selected_].audio)
    {
        ImGui::Spacing();
        ImGui::TextWrapped("Your audio, ready to explore.");
        ImGui::PushTextWrapPos(0);
        ImGui::TextDisabled("Import an audio file to begin.");
        ImGui::PopTextWrapPos();
        return;
    }
    auto& clip = clips_[selected_];
    ImGui::PushFont(nullptr, ImGui::GetFontSize() * 1.18f);
    ImGui::TextWrapped("%s", clip.name.c_str());
    ImGui::PopFont();
    ImGui::TextDisabled("Mono  /  16 kHz  /  %.1f seconds", clip.audio->Duration());
    ImGui::Spacing();
    const float width = std::max(40.f, ImGui::GetContentRegionAvail().x);
    const float transport_height = ImGui::GetFrameHeightWithSpacing() * 2 + ImGui::GetStyle().ItemSpacing.y;
    const float wave_height = std::clamp(ImGui::GetContentRegionAvail().y - transport_height,
                                         ImGui::GetTextLineHeight() * 2, ImGui::GetTextLineHeight() * 10);
    const auto origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("Waveform", ImVec2(width, wave_height));
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + wave_height), ImGui::GetColorU32(ImGuiCol_FrameBg),
                        8);
    for (size_t i = 0; i < clip.peaks.size(); ++i)
    {
        float x = origin.x + static_cast<float>(i) * width / static_cast<float>(clip.peaks.size());
        float h = std::max(1.f, clip.peaks[i] * wave_height * .4f);
        draw->AddLine(ImVec2(x, origin.y + wave_height / 2 - h), ImVec2(x, origin.y + wave_height / 2 + h),
                      ImGui::GetColorU32(ImGuiCol_SliderGrab));
    }
    float x = origin.x + static_cast<float>(position_ / clip.audio->Duration()) * width;
    draw->AddLine(ImVec2(x, origin.y), ImVec2(x, origin.y + wave_height), ImGui::GetColorU32(ImGuiCol_Text), 2);
    if (ImGui::IsItemActive())
        Seek(std::clamp(double(ImGui::GetIO().MousePos.x - origin.x) / width, 0., 1.) * clip.audio->Duration());
    if (ImGui::Button(device_.Playing() ? "Pause" : "Play", ImVec2(ImGui::GetFontSize() * 5, 0)))
    {
        try
        {
            if (device_.Playing())
            {
                position_ = device_.Position();
                device_.Pause();
            }
            else
            {
                if (position_ >= clip.audio->Duration())
                    position_ = 0;
                device_.Play(clip.audio, position_);
            }
        }
        catch (const std::exception& e)
        {
            error_ = e.what();
        }
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%.1f / %.1f s", position_, clip.audio->Duration());
#ifndef DIN_AUDIO_WITH_ASR
    ImGui::TextWrapped("Audio-only build. Native ASR is enabled by building from the repository root.");
#endif
    ImGui::BeginDisabled(clip.Pending()
#ifndef DIN_AUDIO_WITH_ASR
                         || true
#endif
    );
    if (ImGui::Button("Transcribe selected"))
        TranscribeSelected();
    ImGui::EndDisabled();
}
void Workbench::DrawProgress(size_t index)
{
    const auto progress = worker_.Progress();
    if (!progress || progress->index != index)
    {
        ImGui::TextDisabled("Queued / waiting for the current job");
        return;
    }
    const auto& p = *progress;
    const auto now = std::chrono::steady_clock::now();
    const bool running = p.event.stage == din::common::ProgressStage::Transcribing;
    const double elapsed =
        std::chrono::duration<double>(now - (running ? p.inference_started : p.stage_started)).count();
    ImGui::Text("%s  /  %.1f s", StageName(p.event.stage), elapsed);
    ImGui::PushTextWrapPos(0);
    ImGui::TextDisabled("%s", p.event.detail.c_str());
    ImGui::PopTextWrapPos();
    const bool measured = running && p.event.completed_audio_seconds > 0 && p.event.total_audio_seconds > 0;
    if (measured)
    {
        const float fraction =
            static_cast<float>(std::clamp(p.event.completed_audio_seconds / p.event.total_audio_seconds, 0., 1.));
        ImGui::ProgressBar(fraction, ImVec2(-1, 6), "");
        ImGui::Text("%.1f / %.1f s audio", p.event.completed_audio_seconds, p.event.total_audio_seconds);
        if (p.measured_seconds > 0)
            ImGui::Text("%.2fx real time  /  RTF %.3f", p.event.completed_audio_seconds / p.measured_seconds,
                        p.measured_seconds / p.event.completed_audio_seconds);
        ImGui::TextDisabled("Measured at last completed chunk");
    }
    else
    {
        ImGui::ProgressBar(-static_cast<float>(ImGui::GetTime()) * .5f, ImVec2(-1, 6), "");
        if (running)
            ImGui::TextWrapped("Speed will appear when audio has been processed.");
    }
    ImGui::Spacing();
}
void Workbench::Transcript()
{
    if (selected_ < 0)
    {
        ImGui::TextDisabled("Transcripts will appear here.");
        return;
    }
    const auto& clip = clips_[selected_];
    const auto& result = clip.result;
    if (clip.Pending())
        DrawProgress(static_cast<size_t>(selected_));
    if (!result.warning.empty())
        ImGui::TextWrapped("%s", result.warning.c_str());
    if (result.model.empty())
    {
        if (!clip.Pending())
            ImGui::TextWrapped("%s", clip.Status().c_str());
        return;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(ImGui::GetStyle().ItemSpacing.x, 6 * ImGui::GetFontSize() / 17.f));
    ImGui::TextDisabled("%s  /  %.2f s inference", result.model.c_str(), result.seconds);
    if (result.seconds > 0 && result.audio_seconds > 0)
        ImGui::Text("%.2fx real time  /  RTF %.3f", result.audio_seconds / result.seconds,
                    result.seconds / result.audio_seconds);
    if (result.setup_seconds >= .01)
        ImGui::TextDisabled("Model setup %.2f s", result.setup_seconds);
    ImGui::Spacing();
    if (!result.timings.empty())
    {
        const char* timing_label = result.timing_kind == "word"      ? "Word timing"
                                   : result.timing_kind == "segment" ? "Segment timing"
                                                                     : "Token timing";
        const char* options[] = {"Reading", timing_label};
        ImGui::SetNextItemWidth(160);
        ImGui::Combo("##view", &view_, options, 2);
    }
    if (ImGui::Button("Copy text"))
        ImGui::SetClipboardText(result.text.c_str());
    ImGui::SameLine();
    if (ImGui::Button("Export..."))
        ImGui::OpenPopup("Export transcript");
    if (ImGui::BeginPopup("Export transcript"))
    {
        if (ImGui::MenuItem("Plain text (.txt)"))
            OpenDialog(selected_ * 8 + 3);
        if (ImGui::MenuItem("Text and timing (.json)"))
            OpenDialog(selected_ * 8 + 4);
        ImGui::EndPopup();
    }
    ImGui::Dummy(ImVec2(0, ImGui::GetFontSize() * .5f));
    if (view_ == 0 || result.timings.empty())
    {
        ImGui::PushFont(nullptr, ImGui::GetFontSize() * 1.25f);
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(result.text.empty() ? "(No speech recognized)" : result.text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
    }
    else
    {
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(result.timings.size()));
        while (clipper.Step())
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
            {
                const auto& t = result.timings[i];
                ImGui::PushID(static_cast<int>(i));
                std::ostringstream label;
                label << std::fixed << std::setprecision(2) << t.start;
                if (t.end)
                    label << " - " << *t.end;
                label << " s   " << t.text;
                if (ImGui::Selectable(label.str().c_str(),
                                      position_ >= t.start && position_ < t.end.value_or(t.start + .15)))
                    Seek(t.start);
                ImGui::PopID();
            }
    }
    ImGui::PopStyleVar();
}
void Workbench::Export(size_t index, const std::filesystem::path& path, bool json)
{
    const auto& c = clips_.at(index);
    std::string text;
    if (json)
    {
        const auto& r = c.result;
        nlohmann::json document = {{"source", c.name},
                                   {"model", r.model},
                                   {"audio_seconds", r.audio_seconds},
                                   {"transcription_seconds", r.seconds},
                                   {"setup_seconds", r.setup_seconds},
                                   {"text", r.text},
                                   {"warning", r.warning},
                                   {"timing_kind", r.timing_kind},
                                   {"timings", nlohmann::json::array()}};
        for (const auto& t : r.timings)
            document["timings"].push_back({{"text", t.text},
                                           {"start", t.start},
                                           {"end", t.end ? nlohmann::json(*t.end) : nlohmann::json(nullptr)}});
        text = document.dump(2);
    }
    else
        text = c.result.text;
    text += '\n';
    WriteFileAtomically(path, text);
    export_directory_ = std::filesystem::absolute(path).parent_path();
}
void Workbench::Timeline(float height)
{
    const auto& clip = clips_[selected_];
    const double duration = clip.audio->Duration();
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(ImGui::GetStyle().ItemSpacing.x, 6 * ImGui::GetFontSize() / 17.f));
    ImGui::BeginChild("Text timeline", ImVec2(0, height), ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Text timeline");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    const bool zoom_changed =
        ImGui::SliderFloat("Zoom", &timeline_zoom_, 1, std::max(20.f, static_cast<float>(duration / 2)), "%.1fx",
                           ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine();
    if (ImGui::Button("Fit"))
    {
        timeline_zoom_ = 1;
        timeline_start_ = 0;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Follow playback", &follow_playhead_);
    const double span = duration / timeline_zoom_;
    if (zoom_changed)
        timeline_start_ = static_cast<float>(position_ - span * .2);
    timeline_start_ = std::clamp(timeline_start_, 0.f, static_cast<float>(duration - span));
    if (timeline_zoom_ > 1)
    {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(std::max(80.f, ImGui::GetContentRegionAvail().x - 65));
        if (ImGui::SliderFloat("Start", &timeline_start_, 0, static_cast<float>(duration - span), "%.1f s"))
            follow_playhead_ = false;
    }
    if (follow_playhead_)
        timeline_start_ = static_cast<float>(FollowTimeline(timeline_start_, span, duration, position_));
    const auto origin = ImGui::GetCursorScreenPos();
    const float width = std::max(1.f, ImGui::GetContentRegionAvail().x);
    const float lane_top = ImGui::GetTextLineHeight() + 9;
    const float lane_height = ImGui::GetTextLineHeight() + 10;
    ImGui::InvisibleButton("##timecanvas", ImVec2(width, lane_top + lane_height));
    auto* draw = ImGui::GetWindowDrawList();
    auto x_at = [&](double seconds)
    {
        return origin.x + static_cast<float>((seconds - timeline_start_) / span) * width;
    };
    draw->PushClipRect(origin, ImVec2(origin.x + width, origin.y + lane_top + lane_height), true);
    draw->AddRectFilled(ImVec2(origin.x, origin.y + lane_top),
                        ImVec2(origin.x + width, origin.y + lane_top + lane_height),
                        ImGui::GetColorU32(ImGuiCol_FrameBg), 4);
    for (int i = 0; i <= 8; ++i)
    {
        const double seconds = timeline_start_ + span * i / 8.;
        const float x = x_at(seconds);
        char label[32];
        snprintf(label, sizeof(label), "%.1f s", seconds);
        const float label_width = ImGui::CalcTextSize(label).x;
        draw->AddText(ImVec2(std::min(x, origin.x + width - label_width), origin.y),
                      ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
    }
    const Timing* hovered = nullptr;
    const auto mouse = ImGui::GetIO().MousePos;
    for (const auto& t : clip.result.timings)
    {
        const float x = x_at(t.start), end = t.end ? x_at(*t.end) : x + 3;
        if (end < origin.x || x > origin.x + width)
            continue;
        const bool active = t.end && position_ >= t.start && position_ < *t.end;
        draw->AddRectFilled(ImVec2(x, origin.y + lane_top + 2),
                            ImVec2(std::max(x + 2, end - 1), origin.y + lane_top + lane_height - 2),
                            ImGui::GetColorU32(active ? ImGuiCol_HeaderActive : ImGuiCol_Header), 3);
        if (!t.end)
            draw->AddLine(ImVec2(x, origin.y + lane_top), ImVec2(x, origin.y + lane_top + lane_height),
                          ImGui::GetColorU32(ImGuiCol_SliderGrab), 2);
        if (t.end && end - x > ImGui::CalcTextSize(t.text.c_str()).x + 10)
        {
            draw->PushClipRect(ImVec2(std::max(origin.x, x + 4), origin.y + lane_top),
                               ImVec2(std::min(origin.x + width, end - 3), origin.y + lane_top + lane_height), true);
            draw->AddText(ImVec2(x + 5, origin.y + lane_top + 7), ImGui::GetColorU32(ImGuiCol_Text), t.text.c_str());
            draw->PopClipRect();
        }
        if (ImGui::IsItemHovered() && mouse.y >= origin.y + lane_top && mouse.x >= x - 2 &&
            mouse.x <= std::max(x + 4, end))
            hovered = &t;
    }
    if (clip.result.timings.empty())
        draw->AddText(ImVec2(origin.x + 10, origin.y + lane_top + 7), ImGui::GetColorU32(ImGuiCol_TextDisabled),
                      clip.result.model.empty() ? "Timing appears after transcription, when available."
                                                : "No timing data in this result.");
    const float playhead = x_at(position_);
    draw->AddLine(ImVec2(playhead, origin.y + lane_top), ImVec2(playhead, origin.y + lane_top + lane_height),
                  ImGui::GetColorU32(ImGuiCol_Text), 2);
    draw->PopClipRect();
    if (hovered)
    {
        if (hovered->end)
            ImGui::SetTooltip("%s\n%.2f - %.2f s", hovered->text.c_str(), hovered->start, *hovered->end);
        else
            ImGui::SetTooltip("%s\nStart: %.2f s", hovered->text.c_str(), hovered->start);
    }
    if (ImGui::IsItemClicked())
        Seek(hovered ? hovered->start
                     : std::clamp(timeline_start_ + (mouse.x - origin.x) / width * span, 0., duration));
    if (!clip.result.timings.empty())
    {
        // Full labels are deliberately independent of duration: brief words remain readable.
        // The proportional timing bars above remain the source of exact time geometry.
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(5, 0));
        ImGui::BeginChild("Readable words", ImVec2(0, ImGui::GetFrameHeight() + ImGui::GetStyle().ScrollbarSize + 2), 0,
                          ImGuiWindowFlags_HorizontalScrollbar);
        if (ImGui::IsWindowHovered() &&
            (ImGui::GetIO().MouseWheelH != 0 || (ImGui::GetIO().KeyShift && ImGui::GetIO().MouseWheel != 0) ||
             (ImGui::IsMouseDragging(ImGuiMouseButton_Left) &&
              ImGui::GetIO().MousePos.y >=
                  ImGui::GetWindowPos().y + ImGui::GetWindowHeight() - ImGui::GetStyle().ScrollbarSize)))
            follow_playhead_ = false;
        const auto& timings = clip.result.timings;
        if (timing_offsets_.size() != timings.size() + 1 || timing_font_size_ != ImGui::GetFontSize())
        {
            timing_font_size_ = ImGui::GetFontSize();
            timing_offsets_.assign(1, 0);
            for (const auto& t : timings)
                timing_offsets_.push_back(
                    timing_offsets_.back() +
                    ImGui::CalcTextSize(t.text.empty() ? "(token)" : t.text.c_str(), nullptr, true).x +
                    ImGui::GetStyle().FramePadding.x * 2 + ImGui::GetStyle().ItemSpacing.x);
        }
        const auto focus_it = std::upper_bound(timings.begin(), timings.end(), position_,
                                               [](double time, const Timing& t)
                                               {
                                                   return time < t.start;
                                               });
        const size_t focus = focus_it == timings.begin() ? 0 : size_t(focus_it - timings.begin() - 1);
        const auto base = ImGui::GetCursorPos();
        float scroll = ImGui::GetScrollX();
        const float viewport_width = ImGui::GetWindowWidth();
        if (follow_playhead_ &&
            (base.x + timing_offsets_[focus] < scroll || base.x + timing_offsets_[focus + 1] > scroll + viewport_width))
        {
            scroll = std::max(0.f, base.x + timing_offsets_[focus] - viewport_width * .25f);
            ImGui::SetScrollX(scroll);
        }
        auto first = std::upper_bound(timing_offsets_.begin(), timing_offsets_.end(), scroll - base.x);
        size_t first_index = first == timing_offsets_.begin() ? 0 : size_t(first - timing_offsets_.begin() - 1);
        for (size_t i = first_index; i < timings.size() && base.x + timing_offsets_[i] <= scroll + viewport_width; ++i)
        {
            const auto& t = clip.result.timings[i];
            ImGui::SetCursorPos({base.x + timing_offsets_[i], base.y});
            ImGui::PushID(static_cast<int>(i));
            const bool active = position_ >= t.start && (t.end ? position_ < *t.end : i == focus);
            if (active)
                ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
            if (ImGui::Button(t.text.empty() ? "(token)" : t.text.c_str()))
                Seek(t.start);
            if (active)
                ImGui::PopStyleColor();
            if (ImGui::IsItemHovered())
            {
                if (t.end)
                    ImGui::SetTooltip("%.2f - %.2f s", t.start, *t.end);
                else
                    ImGui::SetTooltip("Start: %.2f s", t.start);
            }
            ImGui::PopID();
        }
        ImGui::SetCursorPos({base.x + timing_offsets_.back(), base.y});
        ImGui::Dummy({1, ImGui::GetFrameHeight()});
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
}
void Workbench::Update()
{
    for (auto& c : worker_.Poll())
    {
        auto& clip = clips_.at(c.index);

        if (!c.error.empty())
        {
            clip.state = ClipState::Error;
            clip.error = c.error;
        }
        else if (c.importing)
        {
            clip.audio = std::move(c.audio);
            clip.peaks = std::move(c.peaks);
            clip.state = ClipState::Ready;
        }
        else
        {
            clip.result = std::move(c.result);
            timing_offsets_.clear();
            clip.state = ClipState::Complete;
        }
    }
    const bool playing = device_.Playing();
    if (playing || audio_was_playing_)
        position_ = device_.Position();
    audio_was_playing_ = playing;
}
void Workbench::Draw()
{
    auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("DIN Audio", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    const float unit = ImGui::GetFontSize() / 17.f;
    const float header_y = ImGui::GetCursorPosY();
    const float logo_size = (viewport->WorkSize.y < 700 * unit ? 32 : 56) * unit;
    ImGui::BeginGroup();
    DrawLogo(logo_size);
    ImGui::SameLine();
    ImGui::SetCursorPosY(header_y + (logo_size - ImGui::GetFontSize() * 1.5f) / 2);
    ImGui::PushFont(nullptr, ImGui::GetFontSize() * 1.5f);
    ImGui::TextUnformatted("Audio");
    ImGui::PopFont();
    ImGui::EndGroup();
    ImGui::Dummy(ImVec2(0, 4 * unit));
    if (ImGui::BeginTable("Settings", 4, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_NoPadOuterX))
    {
        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthFixed, 170 * unit);
        ImGui::TableSetupColumn("Execution", ImGuiTableColumnFlags_WidthFixed, 150 * unit);
        ImGui::TableSetupColumn("Model folder", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Options", ImGuiTableColumnFlags_WidthFixed, 100 * unit);
        ImGui::TableNextRow();
        for (const auto* label : {"Model", "Execution", "Model folder", "Options"})
        {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", label);
        }
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("##model", &model_, Models, 4);
        auto& option = options_[model_];
        ImGui::TableSetColumnIndex(1);
        if (model_ == 0)
        {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("TensorRT RTX");
        }
        else
        {
            ImGui::SetNextItemWidth(-1);
            int provider = option.provider == "trt-rtx";
            if (ImGui::Combo("##execution", &provider, "CPU\0TensorRT RTX\0"))
                option.provider = provider ? "trt-rtx" : "cpu";
        }
        ImGui::TableSetColumnIndex(2);
        const float folder_width = ImGui::CalcTextSize("Browse...").x + 2 * ImGui::GetStyle().FramePadding.x;
        ImGui::SetNextItemWidth(
            std::max(50.f, ImGui::GetContentRegionAvail().x - folder_width - ImGui::GetStyle().ItemSpacing.x));
        ImGui::InputText("##modeldir", &option.directory);
        ImGui::SameLine();
        if (ImGui::Button("Browse..."))
            OpenDialog(model_ * 8 + 2);
        ImGui::TableSetColumnIndex(3);
        ModelSettings();
        ImGui::EndTable();
    }
    ImGui::Dummy(ImVec2(0, 4 * unit));
    const bool has_audio = selected_ >= 0 && clips_[selected_].audio;
    const float timeline_height = has_audio ? 190 * unit : 0;
    const float height =
        std::max(100.f, ImGui::GetContentRegionAvail().y - timeline_height - ImGui::GetStyle().ItemSpacing.y * 2);
    if (ImGui::BeginTable("Panels", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoPadOuterX))
    {
        ImGui::TableSetupColumn("Audio library", ImGuiTableColumnFlags_WidthFixed, 210 * unit);
        ImGui::TableSetupColumn("Listen", ImGuiTableColumnFlags_WidthStretch, 1.15f);
        ImGui::TableSetupColumn("Transcript", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableNextRow();
        for (int panel = 0; panel < 3; ++panel)
        {
            ImGui::TableSetColumnIndex(panel);
            ImGui::PushID(panel);
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                                ImVec2(ImGui::GetStyle().ItemSpacing.x,
                                       height < 300 * unit ? 6 * unit : ImGui::GetStyle().ItemSpacing.y));
            ImGui::BeginChild("Panel", ImVec2(0, height), ImGuiChildFlags_AlwaysUseWindowPadding);
            const char* headings[] = {"Audio library", "Listen", "Transcript"};
            ImGui::TextDisabled("%s", headings[panel]);
            try
            {
                if (panel == 0)
                    Library();
                else if (panel == 1)
                    Player();
                else
                    Transcript();
            }
            catch (const std::exception& e)
            {
                error_ = e.what();
            }
            ImGui::EndChild();
            ImGui::PopStyleVar();
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (has_audio)
        Timeline(timeline_height);
    SavePreferences();
    if (!error_.empty() && !ImGui::IsPopupOpen("Audio error"))
        ImGui::OpenPopup("Audio error");
    ImGui::SetNextWindowSize(ImVec2(460 * unit, 0), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Audio error", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextWrapped("%s", error_.c_str());
        if (ImGui::Button("Dismiss"))
        {
            error_.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::End();
}
}  // namespace din::studio
