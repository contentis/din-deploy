// SPDX-License-Identifier: Apache-2.0
#include "preferences.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "paths.h"
namespace din::studio
{
Preferences ReadPreferences(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
        return {};
    if (std::filesystem::file_size(path) > 65536)
        throw std::runtime_error("Preferences file is too large");
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot read preferences");
    Preferences result;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty())
            continue;
        std::istringstream row(line);
        std::string key, value;
        if (!(row >> key >> std::quoted(value)) || !(row >> std::ws).eof())
            throw std::runtime_error("Invalid preferences file");
        result[key] = value;
    }
    if (result["version"] != "1")
        throw std::runtime_error("Unsupported preferences version");
    return result;
}
std::string EncodePreferences(const Preferences& values)
{
    std::ostringstream out;
    for (const auto& [key, value] : values)
        out << key << ' ' << std::quoted(value) << '\n';
    return out.str();
}
double FollowTimeline(double start, double span, double duration, double position)
{
    if (position < start)
        start = position - span * .2;
    else if (position > start + span * .8)
        start = position - span * .8;
    return std::clamp(start, 0., std::max(0., duration - span));
}
void VerifyPreferences(const std::filesystem::path& path)
{
    struct Cleanup
    {
        std::filesystem::path path;
        ~Cleanup()
        {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{path};
    Preferences values = {{"version", "1"},
                          {"model.0.directory", "D:\\models\\speech model"},
                          {"model.0.aligner", "模型/aligner"},
                          {"language", "en"}};
    WriteFileAtomically(path, EncodePreferences(values));
    if (ReadPreferences(path) != values)
        throw std::runtime_error("Preferences round trip failed");
    values["language"] = "de";
    WriteFileAtomically(path, EncodePreferences(values));
    if (ReadPreferences(path) != values)
        throw std::runtime_error("Preferences replacement failed");
    WriteFileAtomically(path, "invalid data\n");
    bool rejected = false;
    try
    {
        ReadPreferences(path);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("Malformed preferences accepted");
    for (double position : {0., 5., 18., 44., 99., 100., 2.})
    {
        const double start = FollowTimeline(20, 10, 100, position);
        if (start < 0 || start > 90 || position < start || position > start + 10)
            throw std::runtime_error("Timeline lost playhead");
    }
}
}  // namespace din::studio
