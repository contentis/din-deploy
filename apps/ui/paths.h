// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <filesystem>
#include <string>
#include <string_view>

namespace din::studio
{
// Replace only after the complete contents have been written beside the destination.
void WriteFileAtomically(const std::filesystem::path& path, std::string_view text);

inline std::filesystem::path Utf8Path(std::string_view text)
{
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(text.data()), text.size()));
}
inline std::string Utf8(const std::filesystem::path& path)
{
    const auto text = path.u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}
}  // namespace din::studio
