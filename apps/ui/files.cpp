// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <stdexcept>

#include "paths.h"
#include <SDL3/SDL.h>

namespace din::studio
{
void WriteFileAtomically(const std::filesystem::path& path, std::string_view text)
{
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    // Reserve a unique directory atomically, without overwriting a neighbour's .tmp file.
    std::filesystem::path folder;
    do
    {
        folder = path.parent_path() / (".din-write-" + std::to_string(SDL_GetTicksNS()));
    } while (!std::filesystem::create_directory(folder));
    struct Cleanup
    {
        std::filesystem::path folder;
        ~Cleanup()
        {
            std::error_code ec;
            std::filesystem::remove(folder / "data", ec);
            std::filesystem::remove(folder, ec);
        }
    } cleanup{folder};
    const auto pending = folder / "data";
    std::ofstream out(pending, std::ios::binary);
    out << text;
    out.close();
    if (!out)
        throw std::runtime_error("Cannot write file");
    if (!SDL_RenamePath(Utf8(pending).c_str(), Utf8(path).c_str()))
        throw std::runtime_error(std::string("Cannot replace file: ") + SDL_GetError());
}
}  // namespace din::studio
