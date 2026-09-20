// SPDX-License-Identifier: Apache-2.0
#include "imgui.h"
#include "imgui_support.h"
#include <SDL3/SDL.h>

namespace din::studio
{
namespace
{
// DIN mark in a 256-unit square. The D bowl is a strip of convex quads;
// its counter and the surrounding background stay transparent.
constexpr ImVec2 Quads[][4] = {
    {{39, 44}, {218, 44}, {218, 52}, {39, 52}},     {{39, 191}, {218, 191}, {218, 199}, {39, 199}},
    {{56, 70}, {74, 70}, {74, 176}, {56, 176}},     {{115, 70}, {133, 70}, {133, 176}, {115, 176}},
    {{140, 70}, {158, 70}, {158, 176}, {140, 176}}, {{140, 70}, {158, 70}, {200, 176}, {182, 176}},
    {{182, 70}, {200, 70}, {200, 176}, {182, 176}},
};
// Outer/inner edges of the D bowl. It overlaps the stem so antialiasing
// only touches the letter's outline, never internal tessellation seams.
constexpr ImVec2 Bowl[][2] = {
    {{65, 70}, {65, 87}},   {{83, 70}, {83, 87}},    {{96, 74}, {88, 89}},    {{104, 84}, {90, 95}},
    {{108, 98}, {90, 98}},  {{108, 148}, {90, 148}}, {{104, 162}, {90, 151}}, {{96, 172}, {88, 157}},
    {{83, 176}, {83, 159}}, {{65, 176}, {65, 159}},
};
}  // namespace

void DrawLogo(float size)
{
    const auto origin = ImGui::GetCursorScreenPos();
    const float scale = size / 256;
    const auto ink = ImGui::GetColorU32(ImGuiCol_Text);
    auto* draw = ImGui::GetWindowDrawList();
    for (const auto& quad : Quads)
    {
        ImVec2 p[4];
        for (int i = 0; i < 4; ++i)
            p[i] = {origin.x + quad[i].x * scale, origin.y + quad[i].y * scale};
        draw->AddQuadFilled(p[0], p[1], p[2], p[3], ink);
    }
    ImVec2 bowl[20];
    for (int i = 0; i < 10; ++i)
    {
        bowl[i] = {origin.x + Bowl[i][0].x * scale, origin.y + Bowl[i][0].y * scale};
        bowl[19 - i] = {origin.x + Bowl[i][1].x * scale, origin.y + Bowl[i][1].y * scale};
    }
    draw->AddConcavePolyFilled(bowl, 20, ink);
    constexpr char caption[] = "DO INFERENCE NOW";
    for (int i = 0; caption[i]; ++i)
        draw->AddText(nullptr, 14 * scale, {origin.x + (40 + 11 * i) * scale, origin.y + 204 * scale}, ink, caption + i,
                      caption + i + 1);
    ImGui::Dummy({size, size});
}

void SetLogoIcon(SDL_Window* window)
{
    // Window managers require pixels. Rasterize the mark once, with no UI texture.
    auto* surface = SDL_CreateSurface(256, 256, SDL_PIXELFORMAT_RGBA32);
    if (!surface)
        return;
    if (auto* renderer = SDL_CreateSoftwareRenderer(surface))
    {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
        SDL_RenderClear(renderer);
        constexpr int indices[] = {0, 1, 2, 0, 2, 3};
        auto draw_quad = [&](const ImVec2* quad)
        {
            SDL_Vertex vertices[4]{};
            for (int i = 0; i < 4; ++i)
            {
                vertices[i].position = {quad[i].x, quad[i].y};
                vertices[i].color = {232 / 255.f, 235 / 255.f, 240 / 255.f, 1};
            }
            SDL_RenderGeometry(renderer, nullptr, vertices, 4, indices, 6);
        };
        for (const auto& quad : Quads)
            draw_quad(quad);
        for (int i = 0; i < 9; ++i)
        {
            const ImVec2 quad[] = {Bowl[i][0], Bowl[i + 1][0], Bowl[i + 1][1], Bowl[i][1]};
            draw_quad(quad);
        }
        SDL_RenderPresent(renderer);
        SDL_SetWindowIcon(window, surface);
        SDL_DestroyRenderer(renderer);
    }
    SDL_DestroySurface(surface);
}
}  // namespace din::studio
