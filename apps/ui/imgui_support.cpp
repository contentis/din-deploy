// SPDX-License-Identifier: Apache-2.0
#include "imgui_support.h"

#include <filesystem>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"
#include <SDL3/SDL.h>

namespace din::studio
{
void InitImGui(SDL_Window* window, SDL_Renderer* renderer, float font_size)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;
    for (const auto* path : {"C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                             "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"})
        if (std::filesystem::exists(path) && io.Fonts->AddFontFromFileTTF(path, font_size))
            break;
    if (io.Fonts->Fonts.empty())
        io.Fonts->AddFontDefault();
    ImGui::GetStyle().FontSizeBase = font_size;
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);
}
void ShutdownImGui()
{
    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}
bool SaveScreenshot(SDL_Renderer* renderer, const char* path)
{
    auto* surface = SDL_RenderReadPixels(renderer, nullptr);
    if (!surface)
        return false;
    const bool saved = SDL_SaveBMP(surface, path);
    SDL_DestroySurface(surface);
    return saved;
}
}  // namespace din::studio
