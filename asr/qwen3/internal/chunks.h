// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace din::asr::qwen3::detail
{
struct AudioChunk
{
    size_t begin;
    size_t end;
};

inline float WindowEnergy(std::span<const float> samples, size_t begin)
{
    // Independent FP32 reductions avoid cancellation drift across quiet windows.
    std::array<float, 8> sums{};
    for (size_t i = 0; i < 1600; i += 8)
        for (size_t j = 0; j < 8; ++j)
            sums[j] += std::abs(samples[begin + i + j]);
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
}

// Qwen3-ASR split_audio_into_chunks: quietest 100 ms within +/-5 seconds,
// then the quietest sample inside that window. Keep the first minimum on ties.
inline std::vector<AudioChunk> SplitAudio(std::span<const float> samples, size_t target_seconds = 1200)
{
    const size_t target = target_seconds * 16000;
    constexpr size_t expand = 5 * 16000, window = 1600;
    std::vector<AudioChunk> chunks;
    size_t start = 0;
    while (samples.size() - start > target)
    {
        const auto cut = start + target;
        const auto left = cut > expand ? std::max(start, cut - expand) : start;
        const auto right = std::min(samples.size(), cut + expand);
        size_t boundary = cut;
        if (right - left > window)
        {
            float best = WindowEnergy(samples, left);
            size_t minimum = left;
            for (size_t i = left + 1; i + window <= right; ++i)
            {
                const float sum = WindowEnergy(samples, i);
                if (sum < best)
                {
                    best = sum;
                    minimum = i;
                }
            }
            boundary = minimum;
            for (size_t i = minimum + 1; i < minimum + window; ++i)
                if (std::abs(samples[i]) < std::abs(samples[boundary]))
                    boundary = i;
        }
        boundary = std::clamp(boundary, start + 1, samples.size());
        chunks.push_back({start, boundary});
        start = boundary;
    }
    if (start < samples.size())
        chunks.push_back({start, samples.size()});
    return chunks;
}
}  // namespace din::asr::qwen3::detail
