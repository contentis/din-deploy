// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "audio.h"
namespace din::studio
{
struct Timing
{
    std::string text;
    double start = 0;
    std::optional<double> end;
    int speaker = -1;
};
struct Result
{
    std::string text, model, timing_kind, warning;
    std::vector<Timing> timings;
    std::map<int, std::string> speakers;
    std::vector<Timing> speaker_activity, speaker_turns;
    double seconds = 0;
    double setup_seconds = 0, audio_seconds = 0;
};

// Inputs are actual model text spans with audio boundaries, never proportional
// slices of the final transcript. Alignment commits atomically on success.
using AlignFunction = std::function<std::vector<Timing>(const din::io::Audio&, const std::string&)>;
void AlignTranscript(Result&, const din::io::Audio&, const std::vector<Timing>& spans, const AlignFunction&);
void VerifyTranscriptAlignment();
void BuildSpeakerTurns(Result&);
std::string SpeakerName(const Result&, int speaker);
std::string SpeakerText(const Result&);
}  // namespace din::studio
