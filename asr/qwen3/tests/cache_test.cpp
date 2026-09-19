// SPDX-License-Identifier: Apache-2.0
#include "ort_session.h"

#include <algorithm>
#include <iostream>

// GPU integration probe for the standard ONNX TensorScatter aliasing contract.
// Input fixture: cache [1,8,8192,128], update [1,8,1,128], position [1]; output out.
int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;
    try
    {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "qwen3_cache_test"};
        din::common::RegisterTensorRTRTXProvider(env);
        din::common::ModelProfile profile;
        profile.enable_cuda_graph = true;
        const auto cache_dir = std::filesystem::path(argv[1]).parent_path() / "cache-probe";
        din::common::OrtRunner runner(env, argv[1], "trt-rtx", cache_dir.string(), {cache_dir.string(), {}}, profile);
        using BF16 = Ort::BFloat16_t;
        din::common::TensorBuffer<BF16> cache(runner, {1, 8, 8192, 128}, true, true);
        din::common::TensorBuffer<BF16> update(runner, {1, 8, 1, 128}, true, true);
        din::common::TensorBuffer<int64_t> position(runner, {1}, true, true);
        cache.Fill(BF16(1.f));
        cache.CopyAsyncToDevice();
        Ort::IoBinding binding(runner.session);
        binding.BindInput("cache", cache.BindingValue());
        binding.BindInput("update", update.BindingValue());
        binding.BindInput("position", position.BindingValue());
        binding.BindOutput("out", cache.BindingValue());
        std::vector<BF16> expected(8 * 8192 * 128, BF16(1.f));
        for (int64_t pos : {128, 129, 8191, 0})
        {
            position.Fill(pos);
            update.Fill(BF16(2.f));
            position.CopyAsyncToDevice();
            update.CopyAsyncToDevice();
            runner.session.Run(Ort::RunOptions{}, binding);
            cache.CopyAsyncToHostWithNotification().Sync();
            for (int head = 0; head < 8; ++head)
                std::fill_n(expected.data() + (head * 8192 + pos) * 128, 128, BF16(2.f));
            for (size_t i = 0; i < expected.size(); ++i)
                if (cache.HostData()[i].val != expected[i].val)
                    throw std::runtime_error("In-place KV update changed the wrong cache element");
        }
        std::cout << "TensorScatter in-place cache checks passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
