// SPDX-License-Identifier: Apache-2.0
#pragma once
struct SDL_Window;
struct SDL_Renderer;

namespace din::studio
{
// The caller owns the window/renderer and chooses its style after initialization.
void InitImGui(SDL_Window* window, SDL_Renderer* renderer, float font_size);
void ShutdownImGui();
// Vector header mark and a window icon generated from the same geometry.
void DrawLogo(float size);
void SetLogoIcon(SDL_Window* window);
// Explicit smoke-test readback only; never part of normal rendering.
bool SaveScreenshot(SDL_Renderer* renderer, const char* path);
}  // namespace din::studio
