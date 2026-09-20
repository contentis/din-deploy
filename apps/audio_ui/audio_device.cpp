// SPDX-License-Identifier: Apache-2.0
#define MINIAUDIO_IMPLEMENTATION
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "miniaudio.h"
#include "workbench.h"

namespace din::studio
{
namespace
{
void Check(ma_result result, const char* action)
{
    if (result != MA_SUCCESS)
        throw std::runtime_error(std::string(action) + ": " + ma_result_description(result));
}
}  // namespace
std::shared_ptr<Audio> Decode(const std::filesystem::path& path)
{
    ma_decoder decoder;
    auto config = ma_decoder_config_init(ma_format_f32, 1, 16000);
#ifdef _WIN32
    Check(ma_decoder_init_file_w(path.c_str(), &config, &decoder), "Open audio");
#else
    Check(ma_decoder_init_file(path.c_str(), &config, &decoder), "Open audio");
#endif
    struct Guard
    {
        ma_decoder* p;
        ~Guard()
        {
            ma_decoder_uninit(p);
        }
    } guard{&decoder};
    auto audio = std::make_shared<Audio>();
    audio->sample_rate = 16000;
    float block[16384];
    for (;;)
    {
        ma_uint64 count = 0;
        const auto status = ma_decoder_read_pcm_frames(&decoder, block, 16384, &count);
        if (status != MA_SUCCESS && status != MA_AT_END)
            Check(status, "Decode audio");
        audio->samples.insert(audio->samples.end(), block, block + count);
        if (status == MA_AT_END || count == 0)
            break;
        Check(status, "Decode audio");
    }
    if (audio->samples.empty())
        throw std::runtime_error("The audio file contains no samples");
    return audio;
}
std::vector<float> Peaks(const Audio& audio)
{
    std::vector<float> peaks(512, 0);
    for (size_t i = 0; i < audio.samples.size(); ++i)
    {
        const auto bin = std::min(size_t(511), i * 512 / audio.samples.size());
        peaks[bin] = std::max(peaks[bin], std::abs(audio.samples[i]));
    }
    return peaks;
}
void SaveWave(const std::filesystem::path& path, const Audio& audio)
{
    ma_encoder encoder;
    auto config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, audio.sample_rate);
#ifdef _WIN32
    Check(ma_encoder_init_file_w(path.c_str(), &config, &encoder), "Save recording");
#else
    Check(ma_encoder_init_file(path.c_str(), &config, &encoder), "Save recording");
#endif
    ma_uint64 written = 0;
    auto status = ma_encoder_write_pcm_frames(&encoder, audio.samples.data(), audio.samples.size(), &written);
    ma_encoder_uninit(&encoder);
    Check(status, "Write recording");
    if (written != audio.samples.size())
        throw std::runtime_error("Incomplete recording write");
}
struct AudioDevice::Impl
{
    ma_device playback{};
    bool playback_ready = false;
    std::shared_ptr<const Audio> audio;
    std::atomic<size_t> cursor{0};
    static constexpr size_t NoSeek = std::numeric_limits<size_t>::max();
    std::atomic<size_t> seek{NoSeek};
    std::atomic<bool> playing{false};
    static void Playback(ma_device* device, void* output, const void*, ma_uint32 frames)
    {
        auto& self = *static_cast<Impl*>(device->pUserData);
        auto out = static_cast<float*>(output);
        std::fill_n(out, frames, 0.f);
        if (!self.playing.load())
            return;
        const auto requested = self.seek.exchange(NoSeek);
        auto start = requested == NoSeek ? self.cursor.load() : requested;
        auto n = std::min(size_t(frames), self.audio->samples.size() - start);
        std::copy_n(self.audio->samples.data() + start, n, out);
        self.cursor.store(start + n);
        if (start + n == self.audio->samples.size())
            self.playing.store(false);
    }
    ~Impl()
    {
        if (playback_ready)
            ma_device_uninit(&playback);
    }
};
AudioDevice::AudioDevice()
    : impl_(std::make_unique<Impl>())
{
}
AudioDevice::~AudioDevice() = default;
void AudioDevice::Pause()
{
    if (impl_->playback_ready)
        Check(ma_device_stop(&impl_->playback), "Pause playback");
    impl_->playing.store(false);
}
void AudioDevice::Play(std::shared_ptr<const Audio> audio, double seconds)
{
    Pause();
    if (!impl_->playback_ready)
    {
        auto config = ma_device_config_init(ma_device_type_playback);
        config.playback.format = ma_format_f32;
        config.playback.channels = 1;
        config.sampleRate = 16000;
        config.dataCallback = Impl::Playback;
        config.pUserData = impl_.get();
        Check(ma_device_init(nullptr, &config, &impl_->playback), "Open output device");
        impl_->playback_ready = true;
    }
    impl_->audio = std::move(audio);
    impl_->seek.store(Impl::NoSeek);
    impl_->cursor.store(std::min(impl_->audio->samples.size(), size_t(std::max(0., seconds) * 16000)));
    impl_->playing.store(true);
    auto result = ma_device_start(&impl_->playback);
    if (result != MA_SUCCESS)
        impl_->playing.store(false);
    Check(result, "Start playback");
}
void AudioDevice::Seek(double seconds)
{
    if (impl_->audio)
        impl_->seek.store(std::min(impl_->audio->samples.size(), size_t(std::max(0., seconds) * 16000)));
}
double AudioDevice::Position() const
{
    const auto requested = impl_->seek.load();
    return double(requested == Impl::NoSeek ? impl_->cursor.load() : requested) / 16000.;
}
bool AudioDevice::Playing() const
{
    return impl_->playing.load();
}
}  // namespace din::studio
