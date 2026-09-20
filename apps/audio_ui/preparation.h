// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <future>
#include <thread>

namespace din::studio
{
// One loader, two independent completion signals. Failure of one optional model
// does not prevent preparation of the other. Joining also protects job callbacks.
struct Preparation
{
    std::future<void> alignment, diarization;
    std::jthread thread;

    Preparation(std::function<void()> align, std::function<void()> diarize)
    {
        std::packaged_task<void()> first(std::move(align)), second(std::move(diarize));
        alignment = first.get_future();
        diarization = second.get_future();
        thread = std::jthread(
            [first = std::move(first), second = std::move(second)]() mutable
            {
                first();
                second();
            });
    }
};
}  // namespace din::studio
