// SPDX-License-Identifier: Apache-2.0
#include "text.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace din::asr::qwen3::detail
{
std::string Trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \r\n\t");
    if (first == std::string::npos)
        return {};
    return text.substr(first, text.find_last_not_of(" \r\n\t") - first + 1);
}

// Official detect_and_fix_repetitions operates on Unicode characters, not UTF-8 bytes.
static std::string FixRepetitions(const std::string& text)
{
    std::vector<std::string_view> chars, filtered;
    for (size_t i = 0; i < text.size();)
    {
        const auto c = static_cast<unsigned char>(text[i]);
        const size_t length = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
        chars.emplace_back(text.data() + i, std::min(length, text.size() - i));
        i += length;
    }
    for (size_t i = 0; i < chars.size();)
    {
        size_t end = i + 1;
        while (end < chars.size() && chars[end] == chars[i])
            ++end;
        filtered.insert(filtered.end(), chars.begin() + i, chars.begin() + (end - i > 20 ? i + 1 : end));
        i = end;
    }
    std::string result;
    for (size_t i = 0; i < filtered.size();)
    {
        size_t advance = 1, keep = 1;
        if (filtered.size() - i >= 40)
            for (size_t length = 1; length <= 20 && i + length * 20 <= filtered.size(); ++length)
            {
                size_t end = i + length;
                while (end + length <= filtered.size() &&
                       std::equal(filtered.begin() + i, filtered.begin() + i + length, filtered.begin() + end))
                    end += length;
                if ((end - i) / length >= 20)
                {
                    keep = length;
                    advance = end - i;
                    break;
                }
            }
        for (size_t j = 0; j < keep; ++j)
            result += filtered[i + j];
        i += advance;
    }
    return result;
}

std::pair<std::string, std::string> ParseOutput(const std::string& raw, const std::string& forced_language)
{
    const auto text = FixRepetitions(Trim(raw));
    if (text.empty())
        return {};
    if (!forced_language.empty())
        return {forced_language, text};
    const auto marker = text.find("<asr_text>");
    if (marker == std::string::npos)
        return {{}, Trim(text)};
    auto meta = text.substr(0, marker);
    std::transform(meta.begin(), meta.end(), meta.begin(),
                   [](unsigned char c)
                   {
                       return std::tolower(c);
                   });
    const auto transcript = Trim(text.substr(marker + 10));
    if (meta.find("language none") != std::string::npos)
        return {{}, transcript};
    std::istringstream lines(meta);
    std::string line;
    while (std::getline(lines, line))
    {
        line = Trim(line);
        if (line.starts_with("language "))
        {
            auto language = Trim(line.substr(9));
            if (!language.empty())
                language[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(language[0])));
            return {language, transcript};
        }
    }
    return {{}, transcript};
}

std::vector<std::string> EnglishWords(const std::string& text)
{
    std::vector<std::string> words;
    std::string word;
    std::string cleaned = text;
    // HF drops typographic punctuation, including curly apostrophes (only ASCII ' is kept).
    for (const auto* punctuation : {"\xe2\x80\x93", "\xe2\x80\x94", "\xe2\x80\x98", "\xe2\x80\x99", "\xe2\x80\x9c",
                                    "\xe2\x80\x9d", "\xe2\x80\xa6"})
    {
        size_t pos = 0;
        while ((pos = cleaned.find(punctuation, pos)) != std::string::npos)
            cleaned.erase(pos, 3);
    }
    for (unsigned char c : cleaned)
    {
        if (c >= 128)
            throw std::runtime_error("Native alignment currently requires ASCII English text");
        if (std::isspace(c))
        {
            if (!word.empty())
                words.push_back(std::move(word));
            word.clear();
        }
        else if (std::isalnum(c) || c == '\'')
            word.push_back(static_cast<char>(c));
    }
    if (!word.empty())
        words.push_back(std::move(word));
    return words;
}

// HF Qwen3ASRProcessor's nondecreasing subsequence repair, including its tie rules.
std::vector<int> FixTimestamps(const std::vector<int>& data)
{
    const int n = static_cast<int>(data.size());
    if (n == 0)
        return {};
    std::vector<int> dp(n, 1), parent(n, -1), result = data;
    std::vector<bool> normal(n, false);
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (data[j] <= data[i] && dp[j] + 1 > dp[i])
            {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
    int i = static_cast<int>(std::max_element(dp.begin(), dp.end()) - dp.begin());
    for (; i >= 0; i = parent[i])
        normal[i] = true;
    for (int begin = 0; begin < n;)
    {
        if (normal[begin])
        {
            ++begin;
            continue;
        }
        int end = begin;
        while (end < n && !normal[end])
            ++end;
        for (int k = begin; k < end; ++k)
        {
            if (begin == 0)
                result[k] = result[end];
            else if (end == n)
                result[k] = result[begin - 1];
            else if (end - begin <= 2)
                result[k] = k - begin + 1 <= end - k ? result[begin - 1] : result[end];
            else
                result[k] = static_cast<int>(result[begin - 1] + (result[end] - result[begin - 1]) *
                                                                     float(k - begin + 1) / (end - begin + 1));
        }
        begin = end;
    }
    return result;
}
}  // namespace din::asr::qwen3::detail
