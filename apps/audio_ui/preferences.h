// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <filesystem>
#include <map>
#include <string>
namespace din::studio
{
using Preferences = std::map<std::string, std::string>;
Preferences ReadPreferences(const std::filesystem::path& path);
std::string EncodePreferences(const Preferences& values);
// Keep the playhead inside the visible range without jumping at each word.
double FollowTimeline(double start, double span, double duration, double position);
void VerifyPreferences(const std::filesystem::path& temporary_file);
}  // namespace din::studio
