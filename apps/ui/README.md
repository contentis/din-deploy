# Shared UI support

`din_ui` supplies the pinned SDL/ImGui backends, native file dialogs, UTF-8 path
conversion, atomic file replacement, vector logo, font setup and smoke-test
screenshots. UI samples link this target directly; the helpers do not depend on
ASR or media SDKs.

For another UI, link `din_ui`, call `InitImGui` with its SDL window and renderer,
apply its style, and call `ShutdownImGui` before destroying the renderer.
`DrawLogo(size)` draws the DIN mark directly in ImGui, using the geometry in
`logo.cpp` and the UI font for the caption. `SetLogoIcon(window)` rasterizes that
geometry once for the OS window icon. There are no logo image assets or textures
to load or manage. Screenshot readback is explicit and only used by smoke checks.

Keep application state, event loops, styling and inference scheduling in each
sample. In particular, video's CUDA/Vulkan renderer and coalesced playback/mask
jobs do not fit audio's sequential import/transcription worker. Share concrete
helpers when another sample needs them, rather than adding a general app framework.
