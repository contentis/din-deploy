// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <filesystem>
#include <thread>

namespace din::studio
{
// Relative/missing destinations resolve to the nearest existing absolute folder.
std::filesystem::path DialogFolder(std::filesystem::path path);
void VerifyFileDialogs(const std::filesystem::path& existing_file);
class FileDialogs
{
public:
    ~FileDialogs();
    void Open(void* window, int action, const std::filesystem::path& location);
    bool Busy() const
    {
        return busy_.load();
    }

private:
    std::thread thread_;
    std::atomic<bool> busy_{false};
};
}  // namespace din::studio
