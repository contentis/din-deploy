# DIN Audio

An audio-only Dear ImGui prototype using SDL3 and Vulkan on Windows and Linux,
with the same source for x64 and ARM64. OpenGL and OpenGL ES are disabled at
build time; OpenCL is not used. SDL software rendering is the fallback when
Vulkan cannot initialize. The active renderer is printed on startup.

## Native inference build

Use the repository's normal SDK/toolchain setup and enable the optional target:

```sh
cmake --preset windows-x64 -DDIN_BUILD_AUDIO_UI=ON
cmake --build out/build/windows-x64 --config Release --target din_audio_ui
```

The same option applies to `windows-arm64`, `linux-x64` and `linux-arm64`.
Use a compiler and inference SDKs matching the target architecture. The full
repository still requires its CUDA/TensorRT RTX dependencies, even when selecting
CPU inference in the UI. Rendering with Vulkan does not change the inference
execution provider. Qwen3 currently requires TensorRT RTX; the other adapters
offer CPU and TensorRT RTX.

On Linux install SDL's desktop development dependencies (X11 and/or Wayland,
audio, Vulkan). For Ubuntu, see the package list in
[`audio-ui.yml`](../../.github/workflows/audio-ui.yml). File dialogs use SDL's
native/portal support on Linux. Windows uses the modern Common Item Dialog
for files and folders. Browsing starts at the current model/aligner folder;
imports remember the last audio folder (initially `assets`). Missing paths fall
back to their nearest existing parent. Exports start beside the source audio,
then remember the last export directory. File paths may also be passed as executable arguments or
dropped onto the window.

Run from the repository root to use the default model paths:

```sh
out/build/windows-x64/bin/Release/din_audio_ui.exe assets/sample.wav
```

Choose an exported model folder, import audio, then use **Transcribe selected**.
Models load only after **Transcribe selected** is pressed and stay cached across files.
After the transcription model loads, one background worker prepares the enabled
aligner, then diarizer, while transcription runs. Their preparation status appears
separately from inference progress. Model execution stays sequential; preparation
can still compete for CPU/GPU memory and compute during a cold start.
RTF includes transcription, alignment and diarization execution, excluding only
initial setup and time actually spent waiting for optional model preparation.
Overlapped loading is not subtracted from processing time.
Changing settings affects subsequently submitted jobs; queued jobs retain their
own settings. Imports and inference are serialized on this worker.

## Audio shell without inference SDKs

For UI development and platform checks, build the folder independently:

```sh
cmake -S apps/audio_ui -B build-audio -DCMAKE_BUILD_TYPE=Release
cmake --build build-audio --config Release --target din_audio_ui
ctest --test-dir build-audio -C Release --output-on-failure
```

This build supports import, waveform seeking and playback.
Transcription is visibly disabled. It does not pretend to run models. The
portability workflow builds this shell on all four OS/architecture combinations;
it does not validate inference SDK compatibility or model accuracy there.

## Current behavior

- WAV, MP3 and FLAC are decoded to mono float audio at 16 kHz on the worker.
  The UI and CLI models share `din_audio`, a small miniaudio-backed loader with
  no inference or windowing dependencies. Device playback stays in the UI.
  The prototype holds decoded files in memory; large libraries need sufficient RAM.
- Transcribe one selected file at a time, with success/error states per file. Retry by
  selecting a failed item and transcribing again. Existing results are retained
  if a new attempt fails.
- Microphone capture is not included. Import existing audio files; transcript
  exports remain available separately.
- **Settings...** next to the model folder exposes only options supported by the
  selected adapter: language and output-token limit for Qwen3, language and CPU
  token sampling for Whisper, encoder profile frames for Parakeet, and language
  for Nemotron. Model folders, providers and options are remembered separately
  for each model across launches; there is no settings save button. Selection,
  workspace, transcript view, follow-playback preference and browsing folders
  also persist. Changes save automatically after a short debounce and flush on
  exit. Preferences live in SDL's per-user `DIN/Audio/preferences.ini` directory
  (`%APPDATA%` on Windows; `$XDG_DATA_HOME` or `~/.local/share` on Linux).
  Audio files/results are not restored as a project. Smoke tests use temporary
  state and never change the user's preferences.
- Progress shows audio decoding, model loading, actual EP-context compilation,
  and transcription. Qwen3 and Whisper report completed audio chunks; other
  adapters show elapsed time until completion. Speed is processed audio seconds
  divided by inference seconds; RTF is its inverse. Model setup is measured
  separately. Loading and compilation have indeterminate progress, not estimated
  percentages. Short single-chunk jobs only report speed on completion.
- Reading is always available for completed results. Segment/word/token timing views
  appear only when the actual result contains timestamps. Token durations are
  not presented as subtitle or word boundaries. Nemotron's absent end times stay
  absent in the display and JSON.
- **Optional processing > Forced alignment** is always visible beneath the
  transcription controls. Enable it and choose a Qwen3 aligner export to add word
  timestamps with any ASR model. Its folder and enable switch are saved independently
  of the transcription model. The model and execution-provider dropdowns currently
  offer Qwen3 aligner with CPU (FP32 export) or TensorRT RTX execution.
  Switching it off retains the folder and keeps native
  timing, including Whisper segments. A bundled aligner is suggested when enabling
  the step, but never runs while the step is off. Existing per-model preferences
  migrate from the selected ASR model. Changes apply only to newly submitted jobs.
  The aligner selects its provider independently of the transcription provider and
  uses the native aligner's supported languages. Whisper passes its detected
  language; RNNT models use the configured language, with English for `auto`.
  Whisper transcribes the full recording once, preserving its long-form context.
  Word alignment consumes those native segments; it does not re-transcribe slices.
  RNNT alignment uses the native token/audio mapping for long recordings.
  A failed alignment retains the recognized text and native timing, with a visible
  warning also included in JSON. Qwen's integrated alignment remains in its native
  pipeline; if it fails, the UI retries Qwen transcription without alignment.
  Re-transcribe an existing result to obtain word timing.
  Optional models have separate enable/selection state, with a snapshot per job.
  Future VAD steps can sit alongside alignment and diarization without adding them
  to each ASR model's settings; no generic pipeline framework is required yet.
- **Optional processing > Diarization** enables Nemotron 3 speaker detection,
  with its own model folder and CPU (FP32 export) / TensorRT RTX provider. Both
  optional models are cached independently and run on the background worker.
  Diarization failure retains the transcript and timing with a warning. Its runtime
  is included in final RTF; loading/compilation remain separate setup time.
- Diarized results open in **Speakers** view. Colors identify speakers in that view,
  timing rows and timeline; thin activity lanes retain overlapping speech. A timed
  span receives the speaker with greatest cumulative overlap, or remains unassigned.
  Segment timestamps produce coarse labels; enable alignment for word-level labels.
  Untimed transcripts remain readable, with speaker activity in the timeline.
  **Rename speakers...** edits names for the current recording without rerunning
  inference. Names appear in speaker-view copy and TXT/JSON exports. TXT groups the
  model's timed text by speaker; Reading view and JSON's `text` keep the original
  transcript. JSON also retains stable speaker IDs and overlapping activity spans.
  Names stay with the current session/export; audio sessions are not project files.

- The text timeline displays actual word/token spans or start markers. Click
  a span/marker to seek, use the ruler to seek anywhere, or zoom and pan. Full
  clickable labels appear in a separate readable word/token row below the
  proportional timing bars. **Follow playback** keeps the playhead and current
  word visible; manually panning disables follow until re-enabled. **Fit** returns
  to the full clip. Long clips can zoom down to a two-second window. Untimed
  results explicitly show no timing data; no synthetic segment boundaries.
- Copy transcript, export UTF-8 TXT, or export JSON with timing provenance.
- Three resizable panels share the audio/result state; platform windowing,
  audio devices, inference and UI are separate files. This is a foundation for
  later workspaces, not yet a timeline editor or a plugin system.
  Common SDL/ImGui setup, native dialogs, paths and logo live in
  [`../ui`](../ui/README.md). JSON export uses the repository's nlohmann/json dependency.
- Closing during queued work is blocked until completion; the existing model
  APIs do not provide safe interruption of in-flight inference. No cancel button
  implies immediate GPU cancellation.

## Verification helpers

```sh
din_audio_ui --self-test
din_audio_ui --smoke-test path/to/audio.wav
din_audio_ui --infer-check 0 path/to/qwen3 path/to/audio.wav
din_audio_ui --smoke-asr path/to/qwen3 path/to/audio.wav [path/to/aligner]
din_audio_ui --smoke-settings 0
din_audio_ui --smoke-whisper path/to/whisper path/to/audio.wav [path/to/aligner]
```

The smoke modes render a hidden window and save `audio-ui-smoke.bmp` in the
working directory. `--smoke-asr` exercises the real queue and transcript view.
Model indices for `--infer-check`: Qwen3=0, Whisper=1, Parakeet=2, Nemotron=3.

Local verification: Windows x64 build, Vulkan rendering, audio worker contracts
and Qwen3 transcription. Linux/ARM64 runtime behavior requires testing on those
devices; CI is provided but has not been run
from this workspace.

Cross-model alignment check: `din_audio_ui --infer-check 1 path/to/whisper path/to/audio.wav path/to/qwen-aligner [cpu|trt-rtx]`.

`--alignment-check 1 path/to/whisper path/to/audio.wav path/to/qwen-aligner trt-rtx`
verifies native segments, unchanged text with alignment, word timestamp bounds,
and preservation of native results after an invalid aligner-folder failure.
Use the `DIN_AUDIO_TEST_WHISPER_MODEL`, `DIN_AUDIO_TEST_ALIGNER_MODEL`, and
`DIN_AUDIO_TEST_RECORDING` CMake cache paths to register this as a native CTest.
The shell CI also runs injected partial-alignment and loading-failure checks;
it does not require inference models. Full native inference remains a separate
hardware/model integration check.

Settings are shared between the UI and queued jobs, with snapshots taken on
submission. The aligner has its own cache. Clip lifecycle uses an explicit state,
and long timing lists/word strips submit only visible items. Existing saved
settings retain their format.
