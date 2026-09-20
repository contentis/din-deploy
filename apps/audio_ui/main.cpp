// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include "imgui_support.h"
#include "workbench.h"
#include <SDL3/SDL.h>

namespace
{
void Style(float scale)
{
    ImGui::StyleColorsDark();
    auto& s = ImGui::GetStyle();
    s.WindowPadding = {20, 20};
    s.FramePadding = {10, 7};
    s.ItemSpacing = {12, 12};
    s.CellPadding = {6, 0};
    s.WindowRounding = 10;
    s.ChildRounding = 8;
    s.FrameRounding = 6;
    s.GrabRounding = 6;
    s.WindowBorderSize = 0;
    s.FrameBorderSize = 0;
    s.Colors[ImGuiCol_WindowBg] = {.085f, .09f, .105f, 1};
    s.Colors[ImGuiCol_ChildBg] = {.12f, .125f, .145f, 1};
    s.Colors[ImGuiCol_PopupBg] = {.15f, .16f, .18f, 1};
    s.Colors[ImGuiCol_FrameBg] = {.17f, .18f, .205f, 1};
    s.Colors[ImGuiCol_FrameBgHovered] = {.22f, .24f, .28f, 1};
    s.Colors[ImGuiCol_FrameBgActive] = {.25f, .29f, .36f, 1};
    s.Colors[ImGuiCol_Button] = {.20f, .21f, .24f, 1};
    s.Colors[ImGuiCol_ButtonHovered] = {.27f, .29f, .33f, 1};
    s.Colors[ImGuiCol_ButtonActive] = {.30f, .34f, .41f, 1};
    s.Colors[ImGuiCol_Header] = {.22f, .27f, .36f, 1};
    s.Colors[ImGuiCol_HeaderHovered] = {.25f, .29f, .36f, 1};
    s.Colors[ImGuiCol_HeaderActive] = {.29f, .35f, .45f, 1};
    s.Colors[ImGuiCol_SliderGrab] = {.52f, .65f, .88f, 1};
    s.Colors[ImGuiCol_SliderGrabActive] = {.65f, .77f, .98f, 1};
    s.Colors[ImGuiCol_CheckMark] = {.65f, .77f, .98f, 1};
    s.Colors[ImGuiCol_Text] = {.91f, .92f, .94f, 1};
    s.Colors[ImGuiCol_TextDisabled] = {.62f, .64f, .69f, 1};
    s.Colors[ImGuiCol_Border] = {.24f, .25f, .28f, 1};
    s.Colors[ImGuiCol_Separator] = {.24f, .25f, .28f, 1};
    s.ScaleAllSizes(scale);
    s.FontScaleDpi = scale;
}
}  // namespace
int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--self-test")
        return din::studio::SelfTest();
    if ((argc >= 5 && argc <= 7) &&
        (std::string(argv[1]) == "--infer-check" || std::string(argv[1]) == "--alignment-check"))
    {
        try
        {
            din::studio::Settings settings;
            settings.model = std::stoi(argv[2]);
            settings.directory = argv[3];
            if (argc >= 6 && std::string(argv[5]) != "-")
                settings.aligner_directory = argv[5];
            if (argc == 7)
                settings.provider = argv[6];
            const auto audio = din::studio::Decode(din::studio::Utf8Path(argv[4]));
            if (std::string(argv[1]) == "--alignment-check")
                return din::studio::AlignmentCheck(settings, *audio);
            din::studio::Inference inference;
            auto result = inference.Run(settings, *audio);
            std::cout << result.text << "\nTiming entries: " << result.timings.size() << '\n';
            if (!result.warning.empty())
                std::cout << "Warning: " << result.warning << '\n';
            if (!settings.aligner_directory.empty() && (result.timing_kind != "word" || result.timings.empty()))
                throw std::runtime_error("Forced alignment did not return word timing");
            for (const auto& timing : result.timings)
            {
                if (!settings.aligner_directory.empty() &&
                    (!timing.end || timing.start < 0 || *timing.end < timing.start))
                    throw std::runtime_error("Invalid alignment interval");
                std::cout << timing.start << " - " << timing.end.value_or(timing.start) << ": " << timing.text << '\n';
            }
            return 0;
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
    const bool smoke_whisper = (argc == 4 || argc == 5) && std::string(argv[1]) == "--smoke-whisper";
    const bool smoke_asr = smoke_whisper || ((argc == 4 || argc == 5) && std::string(argv[1]) == "--smoke-asr");
    const bool smoke_settings = argc == 3 && std::string(argv[1]) == "--smoke-settings";
    bool smoke = smoke_asr || smoke_settings || (argc > 1 && std::string(argv[1]) == "--smoke-test");
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::cerr << SDL_GetError();
        return 1;
    }
    float scale = std::max(1.f, SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay()));
    auto* window =
        SDL_CreateWindow("DIN / Audio", static_cast<int>(1200 * scale), static_cast<int>(760 * scale),
                         SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | (smoke ? SDL_WINDOW_HIDDEN : 0));
    if (!window)
    {
        std::cerr << SDL_GetError();
        SDL_Quit();
        return 1;
    }
    SDL_SetWindowMinimumSize(window, 900, 600);
    // Exercise compact layouts in the existing screenshot smoke check.
    if (smoke)
        if (const auto* size = std::getenv("DIN_AUDIO_SMOKE_SIZE"))
        {
            int width = 0, height = 0;
            if (std::sscanf(size, "%dx%d", &width, &height) == 2 && width >= 900 && height >= 600)
                SDL_SetWindowSize(window, width, height);
        }
    // Never let SDL choose an OpenGL renderer, even if provided by the host.
    auto* renderer = SDL_CreateRenderer(window, "vulkan");
    if (!renderer)
        renderer = SDL_CreateRenderer(window, "software");
    if (!renderer)
    {
        std::cerr << SDL_GetError();
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    std::cout << "Renderer: " << SDL_GetRendererName(renderer) << '\n';
    SDL_SetRenderVSync(renderer, 1);
    din::studio::SetLogoIcon(window);
    din::studio::InitImGui(window, renderer, 17);
    Style(scale);
    int exit_code = 0;
    {
        din::studio::Workbench app(window, !smoke);
        if (smoke_settings)
            app.ShowModelSettings(std::atoi(argv[2]));
        if (smoke_asr)
        {
            app.SelectModel(smoke_whisper ? 1 : 0);
            app.DialogResult(smoke_whisper ? 10 : 2, argv[2]);
            if (argc == 5)
                app.SetTimingModel(argv[4]);
        }
        for (int i = smoke_asr ? 3 : smoke ? 2 : 1; i < (smoke_settings ? 2 : smoke_asr ? 4 : argc); ++i)
            app.Import(din::studio::Utf8Path(argv[i]));
        bool done = false, submitted = false;
        int frames = 0, settled_frames = 0;
        while (!done)
        {
            const auto frame_start = SDL_GetTicks();
            app.Update();
            SDL_Event event;
            while (SDL_PollEvent(&event))
            {
                ImGui_ImplSDL3_ProcessEvent(&event);
                if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
                {
                    if (app.CanClose())
                        done = true;
                    else
                        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "DIN Audio",
                                                 "Let queued work finish before closing.", window);
                }
                if (event.type == SDL_EVENT_DROP_FILE)
                    app.Import(din::studio::Utf8Path(event.drop.data));
                if (event.type == SDL_EVENT_USER && event.user.data1)
                {
                    app.DialogResult(event.user.code, static_cast<char*>(event.user.data1));
                    SDL_free(event.user.data1);
                }
            }
            if (!smoke && (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED))
            {
                SDL_Delay(30);
                continue;
            }
            ImGui_ImplSDLRenderer3_NewFrame();
            ImGui_ImplSDL3_NewFrame();
            ImGui::NewFrame();
            app.Draw();
            ImGui::Render();
            if (smoke_asr && !submitted && app.CanClose())
            {
                app.TranscribeSelected();
                submitted = true;
            }
            SDL_SetRenderDrawColor(renderer, 17, 24, 28, 255);
            SDL_RenderClear(renderer);
            ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
            settled_frames = app.CanClose() ? settled_frames + 1 : 0;
            if (smoke && frames >= 30 && settled_frames >= 3)
            {
                if (smoke_asr && !app.HasTranscript())
                {
                    std::cerr << "Smoke inference did not produce a completed result\n";
                    exit_code = 1;
                }
                if (!din::studio::SaveScreenshot(renderer, "audio-ui-smoke.bmp"))
                {
                    std::cerr << SDL_GetError();
                    exit_code = 1;
                }
                done = true;
            }
            SDL_RenderPresent(renderer);
            ++frames;
            const auto elapsed = SDL_GetTicks() - frame_start;
            if (elapsed < 16)
                SDL_Delay(static_cast<Uint32>(16 - elapsed));
        }
    }
    din::studio::ShutdownImGui();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return exit_code;
}
