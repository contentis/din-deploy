// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <charconv>
#include <stdexcept>

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include "preferences.h"
#include "workbench.h"
#include <SDL3/SDL.h>
namespace din::studio
{
void Workbench::LoadPreferences()
{
    try
    {
        if (preferences_path_.empty())
        {
            char* location = SDL_GetPrefPath("DIN", "Audio");
            if (!location)
                throw std::runtime_error(SDL_GetError());
            preferences_path_ = Utf8Path(location) / "preferences.ini";
            SDL_free(location);
        }
        const auto values = ReadPreferences(preferences_path_);
        auto number = [&](const std::string& key, int fallback, int low, int high)
        {
            auto found = values.find(key);
            if (found == values.end())
                return fallback;
            int result = 0;
            const auto& text = found->second;
            auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
            return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() ? std::clamp(result, low, high)
                                                                                       : fallback;
        };
        auto field = [&](const std::string& key, auto& destination)
        {
            auto found = values.find(key);
            if (found != values.end())
                destination = found->second;
        };
        for (size_t i = 0; i < options_.size(); ++i)
        {
            auto& option = options_[i];
            const auto prefix = "model." + std::to_string(i) + ".";
            field(prefix + "directory", option.directory);
            field(prefix + "language", option.language);
            option.provider = number(prefix + "provider", option.provider == "trt-rtx", 0, 1) ? "trt-rtx" : "cpu";
            option.max_tokens = number(prefix + "max_tokens", option.max_tokens, 1, 32768);
            option.encoder_frames = number(prefix + "encoder_frames", option.encoder_frames, 1, 1048576);
            option.cpu_sampling = number(prefix + "cpu_sampling", option.cpu_sampling, 0, 1) != 0;
        }
        // Qwen previously forced TensorRT regardless of its saved provider.
        if (number("version", 1, 1, 2) < 2)
            options_[0].provider = "trt-rtx";
        model_ = number("selected_model", 0, 0, 3);
        field("step.alignment.directory", aligner_.directory);
        aligner_.provider = number("step.alignment.provider", 1, 0, 1) ? "trt-rtx" : "cpu";
        aligner_.enabled = number("step.alignment.enabled", 0, 0, 1) != 0;
        if (!values.empty() && !values.contains("step.alignment.enabled"))
        {
            // Migrate the previously active model's alignment choice once.
            field("model." + std::to_string(model_) + ".aligner", aligner_.directory);
            const auto bundled = Utf8Path(options_[model_].directory) / "aligner";
            if (aligner_.directory.empty() && std::filesystem::is_regular_file(bundled / "metadata.json"))
                aligner_.directory = Utf8(bundled);
            aligner_.enabled = !aligner_.directory.empty();
            if (aligner_.directory.empty())
                field("model.0.aligner", aligner_.directory);
        }
        field("step.diarization.directory", diarizer_.directory);
        diarizer_.provider = number("step.diarization.provider", 1, 0, 1) ? "trt-rtx" : "cpu";
        diarizer_.enabled = number("step.diarization.enabled", 0, 0, 1) != 0;
        view_ = number("transcript_view", 0, 0, 2);
        follow_playhead_ = number("follow_playhead", 1, 0, 1) != 0;
        timeline_height_ = static_cast<float>(number("timeline_height", 210, 100, 10000));
        if (values.contains("import_directory"))
            import_directory_ = Utf8Path(values.at("import_directory"));
        if (values.contains("export_directory"))
            export_directory_ = Utf8Path(values.at("export_directory"));
    }
    catch (const std::exception& e)
    {
        error_ = std::string("Could not restore settings: ") + e.what();
    }
    pending_preferences_ = PreferencesText();
    saved_preferences_ = std::filesystem::exists(preferences_path_) ? pending_preferences_ : "";
}
std::string Workbench::PreferencesText() const
{
    Preferences values = {
        {"version", "2"},
        {"selected_model", std::to_string(model_)},
        {"transcript_view", std::to_string(view_)},
        {"follow_playhead", std::to_string(follow_playhead_)},
        {"timeline_height", std::to_string(static_cast<int>(timeline_height_))},
        {"step.diarization.enabled", std::to_string(diarizer_.enabled)},
        {"step.diarization.provider", std::to_string(diarizer_.provider == "trt-rtx")},
        {"step.diarization.directory",
         diarizer_.directory.empty() ? "" : Utf8(std::filesystem::absolute(Utf8Path(diarizer_.directory)))},
        {"step.alignment.provider", std::to_string(aligner_.provider == "trt-rtx")},
        {"step.alignment.enabled", std::to_string(aligner_.enabled)},
        {"step.alignment.directory",
         aligner_.directory.empty() ? "" : Utf8(std::filesystem::absolute(Utf8Path(aligner_.directory)))},
        {"import_directory", Utf8(import_directory_)},
        {"export_directory", Utf8(export_directory_)}};
    for (size_t i = 0; i < options_.size(); ++i)
    {
        const auto& option = options_[i];
        const auto prefix = "model." + std::to_string(i) + ".";
        // Resolve model paths when saving so a later launch from another folder works.
        values[prefix + "directory"] =
            !option.directory.empty() ? Utf8(std::filesystem::absolute(Utf8Path(option.directory))) : "";
        values[prefix + "language"] = option.language;
        values[prefix + "provider"] = std::to_string(option.provider == "trt-rtx");
        values[prefix + "max_tokens"] = std::to_string(option.max_tokens);
        values[prefix + "encoder_frames"] = std::to_string(option.encoder_frames);
        values[prefix + "cpu_sampling"] = std::to_string(option.cpu_sampling);
    }
    return EncodePreferences(values);
}
void Workbench::SavePreferences(bool immediately)
{
    if (preferences_path_.empty())
        return;
    try
    {
        const auto snapshot = PreferencesText();
        const auto now = std::chrono::steady_clock::now();
        if (snapshot != pending_preferences_)
        {
            pending_preferences_ = snapshot;
            preferences_changed_ = now;
        }
        if (snapshot == saved_preferences_)
            return;
        if (!immediately && now - preferences_changed_ < std::chrono::milliseconds(700))
            return;
        WriteFileAtomically(preferences_path_, snapshot);
        saved_preferences_ = snapshot;
    }
    catch (const std::exception& e)
    {
        error_ = std::string("Could not save settings: ") + e.what();
        // Avoid retrying a failing disk write on every frame.
        preferences_changed_ = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        if (immediately)
            SDL_Log("%s", error_.c_str());
    }
}
std::filesystem::path Workbench::ExpectedAligner() const
{
    const auto directory = Utf8Path(options_[model_].directory);
    const auto bundled = directory / "aligner";
    std::error_code ec;
    if (std::filesystem::is_regular_file(bundled / "metadata.json", ec))
        return bundled;
    const auto sibling = directory.parent_path() / ("aligner-" + Utf8(directory.filename()));
    if (model_ == 0 || std::filesystem::is_regular_file(sibling / "metadata.json", ec))
        return sibling;
    // Offer the user's existing Qwen aligner to the other ASR models.
    if (!aligner_.directory.empty())
        return Utf8Path(aligner_.directory);
    const auto qwen = Utf8Path(options_[0].directory);
    if (std::filesystem::is_regular_file(qwen / "aligner" / "metadata.json", ec))
        return qwen / "aligner";
    return qwen.parent_path() / ("aligner-" + Utf8(qwen.filename()));
}
void Workbench::OpenDialog(int action)
{
    try
    {
        const int kind = action % 8;
        std::filesystem::path location;
        if (kind == 1)
            location = import_directory_;
        else if (kind == 2)
            location = Utf8Path(options_.at(action / 8).directory);
        else if (kind == 7)
            location = Utf8Path(diarizer_.directory);
        else if (kind == 6)
            location = !aligner_.directory.empty() ? Utf8Path(aligner_.directory) : ExpectedAligner();
        else
        {
            const auto& clip = clips_.at(action / 8);
            location = (export_directory_.empty() ? clip.source.parent_path() : export_directory_) / clip.source.stem();
            location += kind == 3 ? ".txt" : ".json";
        }
        dialogs_.Open(window_, action, location);
    }
    catch (const std::exception& e)
    {
        error_ = e.what();
    }
}
void Workbench::SetTimingModel(const std::string& path)
{
    aligner_.directory = path;
    aligner_.enabled = !path.empty();
}
void Workbench::SetDiarizationModel(const std::string& path)
{
    diarizer_.directory = path;
    diarizer_.enabled = !path.empty();
}
void Workbench::OptionalSteps()
{
    const float unit = ImGui::GetFontSize() / 17.f;
    ImGui::Spacing();
    ImGui::TextDisabled("Optional processing");
    if (!ImGui::BeginTable("Processing steps", 4, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_NoPadOuterX))
        return;
    ImGui::TableSetupColumn("Step", ImGuiTableColumnFlags_WidthFixed, 170 * unit);
    ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthFixed, 190 * unit);
    ImGui::TableSetupColumn("Execution", ImGuiTableColumnFlags_WidthFixed, 150 * unit);
    ImGui::TableSetupColumn("Folder", ImGuiTableColumnFlags_WidthStretch);
    auto row = [&](const char* label, const char* model, OptionalModel& step, int action)
    {
        ImGui::PushID(action);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const auto padding = ImGui::GetStyle().FramePadding;
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + padding.y - unit);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(padding.x, unit));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3 * unit);
        const bool changed = ImGui::Checkbox(label, &step.enabled);
        ImGui::PopStyleVar(2);
        if (changed && action == 6 && step.enabled && step.directory.empty())
        {
            const auto detected = ExpectedAligner();
            std::error_code ec;
            if (std::filesystem::is_regular_file(detected / "metadata.json", ec))
                step.directory = Utf8(detected);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Optional step for new transcriptions. Its model selection is retained when switched off.");
        ImGui::BeginDisabled(!step.enabled);
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##model", model))
        {
            ImGui::Selectable(model, true);
            ImGui::SetItemDefaultFocus();
            ImGui::EndCombo();
        }
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        int provider = step.provider == "trt-rtx";
        if (ImGui::Combo("##ep", &provider, "CPU\0TensorRT RTX\0"))
            step.provider = provider ? "trt-rtx" : "cpu";
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Use an FP32 model export for CPU execution.");
        ImGui::TableNextColumn();
        const float browse_width = ImGui::CalcTextSize("Browse...").x + 2 * ImGui::GetStyle().FramePadding.x;
        ImGui::SetNextItemWidth(
            std::max(50.f, ImGui::GetContentRegionAvail().x - browse_width - ImGui::GetStyle().ItemSpacing.x));
        ImGui::InputTextWithHint("##folder", "Select an exported model folder", &step.directory);
        ImGui::SameLine();
        if (ImGui::Button("Browse..."))
            OpenDialog(action);
        ImGui::EndDisabled();
        ImGui::PopID();
    };
    row("Forced alignment", "Qwen3 aligner", aligner_, 6);
    row("Diarization", "Nemotron 3", diarizer_, 7);
    ImGui::EndTable();
}
void Workbench::ModelSettings()
{
    if (ImGui::Button("Settings...", ImVec2(-1, 0)) || open_model_settings_)
    {
        ImGui::OpenPopup("Model settings");
        open_model_settings_ = false;
    }
    ImGui::SetNextWindowSize(
        ImVec2(std::min(540.f * ImGui::GetFontSize() / 17.f, ImGui::GetMainViewport()->WorkSize.x - 40), 0));
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(.5f, .5f));
    if (!ImGui::BeginPopup("Model settings"))
        return;
    auto& option = options_[model_];
    ImGui::Text("%s settings", Models[model_]);
    ImGui::PushTextWrapPos(0);
    ImGui::TextDisabled("Saved automatically. Changes apply to new jobs.");
    ImGui::Spacing();
    if (model_ != 2)
    {
        ImGui::TextUnformatted("Language");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##language", &option.language);
        ImGui::TextDisabled("auto or a language code supported by the export, e.g. en");
    }
    if (model_ == 0)
    {
        ImGui::TextUnformatted("Maximum output tokens per chunk");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputInt("##tokens", &option.max_tokens, 64, 256))
            option.max_tokens = std::clamp(option.max_tokens, 1, 32768);
    }
    else if (model_ == 1)
    {
        ImGui::BeginDisabled(option.provider == "cpu");
        ImGui::Checkbox("CPU token sampling", &option.cpu_sampling);
        ImGui::EndDisabled();
        ImGui::TextDisabled("For TensorRT RTX: sample decoder tokens on CPU. Native segment timestamps are available "
                            "without a forced aligner.");
    }
    else if (model_ == 2)
    {
        if (ImGui::CollapsingHeader("Advanced"))
        {
            ImGui::TextUnformatted("Encoder profile frames");
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputInt("##frames", &option.encoder_frames, 1024, 8192))
                option.encoder_frames = std::clamp(option.encoder_frames, 1, 1048576);
            ImGui::TextDisabled(
                "TensorRT profile limit. The default (65536) uses the export's metadata when supplied.");
        }
        ImGui::TextDisabled("Native token start times and durations are shown automatically.");
    }
    else
        ImGui::TextDisabled("Native token start markers are shown automatically; end times are not supplied.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    if (ImGui::Button("Done"))
        ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}
void Workbench::SelectModel(int model)
{
    model_ = std::clamp(model, 0, 3);
}
void Workbench::ShowModelSettings(int model)
{
    SelectModel(model);
    open_model_settings_ = true;
}
}  // namespace din::studio
