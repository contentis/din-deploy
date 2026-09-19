// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace din::asr::qwen3::detail
{
std::string Trim(const std::string& text);
std::pair<std::string, std::string> ParseOutput(const std::string& raw, const std::string& forced_language = {});
std::vector<std::string> EnglishWords(const std::string& text);
std::vector<int> FixTimestamps(const std::vector<int>& milliseconds);
}  // namespace din::asr::qwen3::detail
