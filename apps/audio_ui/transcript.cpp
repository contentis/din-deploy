// SPDX-License-Identifier: Apache-2.0
#include "transcript.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace din::studio
{
std::string SpeakerName(const Result& result, int speaker)
{
    if (speaker < 0)
        return "Unassigned";
    const auto found = result.speakers.find(speaker);
    return found != result.speakers.end() && !found->second.empty() ? found->second
                                                                    : "Speaker " + std::to_string(speaker + 1);
}
void BuildSpeakerTurns(Result& result)
{
    result.speaker_turns.clear();
    for (const auto& timing : result.timings)
    {
        if (result.speaker_turns.empty() || result.speaker_turns.back().speaker != timing.speaker)
            result.speaker_turns.push_back(timing);
        else
        {
            auto& turn = result.speaker_turns.back();
            turn.text += " " + timing.text;
            turn.end = timing.end;
        }
    }
}
std::string SpeakerText(const Result& result)
{
    if (result.speakers.empty() || result.speaker_turns.empty())
        return result.text;
    std::string text;
    for (const auto& turn : result.speaker_turns)
    {
        if (!text.empty())
            text += "\n\n";
        text += SpeakerName(result, turn.speaker) + ": " + turn.text;
    }
    return text;
}
void AlignTranscript(Result& result, const din::io::Audio& audio, const std::vector<Timing>& spans,
                     const AlignFunction& align)
{
    if (result.text.empty())
        return;
    try
    {
        if (spans.empty())
            throw std::runtime_error("No audio/text boundaries available for alignment");
        std::vector<Timing> words;
        for (const auto& span : spans)
        {
            if (span.text.empty())
                continue;
            if (!std::isfinite(span.start) || !span.end || !std::isfinite(*span.end) || span.start < 0 ||
                *span.end <= span.start || *span.end > audio.Duration() + .01)
                throw std::runtime_error("Invalid transcription segment boundaries");
            const size_t begin = std::min(audio.samples.size(), size_t(std::floor(span.start * audio.sample_rate)));
            const size_t end = std::min(audio.samples.size(), size_t(std::ceil(*span.end * audio.sample_rate)));
            if (end <= begin || end - begin > size_t(180 * audio.sample_rate))
                throw std::runtime_error("Alignment needs audio/text segments of at most 180 seconds");
            din::io::Audio slice;
            slice.sample_rate = audio.sample_rate;
            slice.samples.assign(audio.samples.begin() + begin, audio.samples.begin() + end);
            auto aligned = align(slice, span.text);
            if (aligned.empty())
                throw std::runtime_error("Aligner returned no word timestamps");
            for (auto& word : aligned)
            {
                if (!std::isfinite(word.start) || !word.end || !std::isfinite(*word.end) || word.start < 0 ||
                    *word.end < word.start || *word.end > slice.Duration() + .1)
                    throw std::runtime_error("Aligner returned invalid word timestamps");
                // Timestamp bins can round slightly beyond the actual last sample.
                word.start = std::min(word.start, slice.Duration()) + double(begin) / audio.sample_rate;
                *word.end = std::min(*word.end, slice.Duration()) + double(begin) / audio.sample_rate;
                words.push_back(std::move(word));
            }
        }
        if (words.empty())
            throw std::runtime_error("Aligner returned no word timestamps");
        result.timings = std::move(words);
        result.timing_kind = "word";
        result.warning.clear();
    }
    catch (const std::exception& e)
    {
        result.warning = std::string("Word alignment unavailable; transcript and native timing retained. ") + e.what();
    }
}
void VerifyTranscriptAlignment()
{
    din::io::Audio audio;
    audio.sample_rate = 16000;
    audio.samples.resize(4 * 16000);
    const std::vector<Timing> spans = {{"First.", 0, 1}, {"Second!", 2, 3}};
    Result native;
    native.text = "First. Second!";
    native.timing_kind = "segment";
    native.timings = spans;
    auto success = native;
    AlignTranscript(success, audio, spans,
                    [](const auto&, const auto& text)
                    {
                        return std::vector<Timing>{{text, .1, .8}};
                    });
    if (success.text != native.text || success.timing_kind != "word" || success.timings.size() != 2 ||
        std::abs(success.timings[1].start - 2.1) > .0001)
        throw std::runtime_error("Alignment changed transcript or lost segment offsets");
    auto failure = native;
    int calls = 0;
    AlignTranscript(failure, audio, spans,
                    [&](const auto&, const auto& text)
                    {
                        if (++calls == 2)
                            throw std::runtime_error("injected alignment failure");
                        return std::vector<Timing>{{text, .1, .8}};
                    });
    if (failure.text != native.text || failure.warning.empty() || failure.timing_kind != "segment" ||
        failure.timings.size() != 2 || failure.timings[0].start != 0 || failure.timings[1].start != 2)
        throw std::runtime_error("Partial alignment failure discarded native results");
    AlignTranscript(failure, audio, spans,
                    [](const auto&, const auto&) -> std::vector<Timing>
                    {
                        throw std::runtime_error("injected model loading failure");
                    });
    if (failure.text != native.text || failure.timings.size() != 2 || failure.warning.empty())
        throw std::runtime_error("Alignment loading failure discarded transcript");
}
}  // namespace din::studio
