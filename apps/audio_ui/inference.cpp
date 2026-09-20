// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "preparation.h"
#include "workbench.h"
#ifdef DIN_AUDIO_WITH_ASR
#include "diarization.h"
#include "nemotron.h"
#include "parakeet_tdt.h"
#include "qwen3.h"
#include "whisper.h"
#endif

namespace din::studio
{
struct Inference::Impl
{
    std::optional<Settings> settings;
    din::common::ProgressCallback progress;
    din::common::ProgressCallback alignment_progress, diarization_progress;
    const din::common::ProgressCallback forward_alignment = [this](const auto& p)
    {
        if (alignment_progress)
            alignment_progress(p);
    };
    const din::common::ProgressCallback forward_diarization = [this](const auto& p)
    {
        if (diarization_progress)
            diarization_progress(p);
    };
    const din::common::ProgressCallback forward_progress = [this](const auto& p)
    {
        if (progress)
            progress(p);
    };
#ifdef DIN_AUDIO_WITH_ASR
    std::unique_ptr<asr::qwen3::Qwen3Pipeline> qwen;
    std::unique_ptr<asr::qwen3::Qwen3ForcedAligner> aligner;
    std::unique_ptr<asr::whisper::WhisperPipeline> whisper;
    std::unique_ptr<asr::parakeet::ParakeetPipeline> parakeet;
    std::unique_ptr<asr::nemotron::NemotronPipeline> nemotron;
    std::unique_ptr<asr::diarization::Pipeline> diarizer;
    std::string diarizer_directory, diarizer_provider;
    std::unique_ptr<io::Tokenizer> tokenizer;
    double frame_rate = 0;
    std::filesystem::path loaded_aligner;
    std::string aligner_provider;
    bool qwen_alignment = true;
    static std::filesystem::path AlignerPath(const Settings& s)
    {
        // Empty explicitly disables alignment, including bundled models.
        return Utf8Path(s.aligner_directory);
    }
    void LoadAligner(const std::filesystem::path& path, const std::string& provider)
    {
        if (aligner && loaded_aligner == path && aligner_provider == provider)
            return;
        aligner.reset();
        loaded_aligner.clear();
        asr::qwen3::ForcedAlignerConfig c;
        c.model_dir = path;
        c.provider = provider;
        c.progress = forward_alignment;
        aligner = std::make_unique<asr::qwen3::Qwen3ForcedAligner>(c);
        loaded_aligner = path;
        aligner_provider = provider;
    }
    void LoadDiarizer(const Settings& s)
    {
        if (diarizer && diarizer_directory == s.diarizer_directory && diarizer_provider == s.diarizer_provider)
            return;
        diarizer.reset();
        asr::diarization::Config c;
        c.model_dir = Utf8Path(s.diarizer_directory);
        c.provider = s.diarizer_provider;
        c.progress = forward_diarization;
        diarizer = std::make_unique<asr::diarization::Pipeline>(c);
        diarizer_directory = s.diarizer_directory;
        diarizer_provider = s.diarizer_provider;
    }
    void Load(const Settings& s, bool enable_qwen_alignment = true)
    {
        auto model_settings = s;
        model_settings.diarizer_directory.clear();
        model_settings.diarizer_provider.clear();
        if (s.model != 0)
        {
            model_settings.aligner_directory.clear();
            model_settings.aligner_provider.clear();
        }
        if (settings && *settings == model_settings && qwen_alignment == enable_qwen_alignment)
            return;
        qwen_alignment = enable_qwen_alignment;
        qwen.reset();
        whisper.reset();
        parakeet.reset();
        nemotron.reset();
        tokenizer.reset();
        settings.reset();
        const auto dir = Utf8Path(s.directory);
        if (!std::filesystem::is_directory(dir))
            throw std::runtime_error("Choose an exported model directory first");
        if (s.model == 0)
        {
            asr::qwen3::Qwen3Config c;
            c.provider = s.provider;
            c.aligner_provider = s.aligner_provider;
            c.progress = forward_progress;
            c.defer_alignment = true;
            c.alignment_progress = forward_alignment;
            c.model_dir = dir;
            c.lang_id = s.language;
            c.max_new_tokens = s.max_tokens;
            if (enable_qwen_alignment)
                c.aligner_dir = AlignerPath(s);
            qwen = std::make_unique<asr::qwen3::Qwen3Pipeline>(c);
        }
        else if (s.model == 1)
        {
            asr::whisper::WhisperConfig c;
            c.progress = forward_progress;
            c.model_dir = dir;
            c.provider = s.provider;
            c.lang_id = s.language;
            c.disable_cuda_sampling = s.cpu_sampling;
            whisper = std::make_unique<asr::whisper::WhisperPipeline>(c);
        }
        else if (s.model == 2)
        {
            asr::parakeet::ParakeetConfig c;
            c.encoder_profile_frames = s.encoder_frames;
            c.progress = forward_progress;
            c.model_dir = dir;
            c.provider = s.provider;
            parakeet = std::make_unique<asr::parakeet::ParakeetPipeline>(c);
            frame_rate = asr::parakeet::Metadata::Load(dir / "metadata.json").FrameRate();
            tokenizer = std::make_unique<io::Tokenizer>((dir / "tokenizer.json").string(), io::TokenizerFormat::Json);
        }
        else
        {
            asr::nemotron::NemotronConfig c;
            c.lang_id = s.language;
            c.progress = forward_progress;
            c.model_dir = dir;
            c.provider = s.provider;
            nemotron = std::make_unique<asr::nemotron::NemotronPipeline>(c);
            frame_rate = asr::nemotron::Metadata::Load(dir / "metadata.json").FrameRate();
            tokenizer = std::make_unique<io::Tokenizer>((dir / "vocab.txt").string(), io::TokenizerFormat::Vocab);
        }
        settings = model_settings;
    }
#endif
};
Inference::Inference()
    : impl_(std::make_unique<Impl>())
{
}
Inference::~Inference() = default;
Result Inference::Run(const Settings& s, const Audio& audio, din::common::ProgressCallback progress,
                      PreparationCallback preparation)
{
    if (s.model < 0 || s.model > 3)
        throw std::runtime_error("Unknown model");
    if (audio.sample_rate != 16000 || audio.samples.empty())
        throw std::runtime_error("Expected nonempty 16 kHz mono audio");
#ifdef DIN_AUDIO_WITH_ASR
    double completed_audio = 0;
    impl_->progress = [&](const din::common::InferenceProgress& event)
    {
        if (event.stage == din::common::ProgressStage::Transcribing)
            completed_audio = std::max(completed_audio, event.completed_audio_seconds);
        if (progress)
            progress(event);
    };
    struct ResetCallback
    {
        Impl& impl;
        ~ResetCallback()
        {
            impl.progress = {};
            impl.alignment_progress = {};
            impl.diarization_progress = {};
        }
    } reset{*impl_};
    const auto setup_start = std::chrono::steady_clock::now();
    if (progress)
        progress({din::common::ProgressStage::LoadingModel, Models[s.model]});
    Result out;
    const auto requested_aligner = Impl::AlignerPath(s);
    impl_->Load(s);
    out.setup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();
    out.model = Models[s.model];
    const auto start = std::chrono::steady_clock::now();
    double optional_setup = 0;
    auto preparing = [&](int step, const std::string& detail)
    {
        if (preparation)
            preparation(step, detail);
    };
    auto loading_progress = [&](int step, const din::common::InferenceProgress& event)
    {
        preparing(step,
                  std::string(event.stage == din::common::ProgressStage::CompilingModel ? "Compiling: " : "Loading: ") +
                      event.detail);
    };
    impl_->alignment_progress = [&](const auto& event)
    {
        loading_progress(0, event);
    };
    impl_->diarization_progress = [&](const auto& event)
    {
        loading_progress(1, event);
    };
    std::unique_ptr<Preparation> loading;
    if (!requested_aligner.empty() || !s.diarizer_directory.empty())
    {
        if (!requested_aligner.empty())
            preparing(0, "Queued");
        if (!s.diarizer_directory.empty())
            preparing(1, "Queued");
        loading = std::make_unique<Preparation>(
            [&]
            {
                if (requested_aligner.empty())
                    return;
                preparing(0, "Loading model");
                try
                {
                    if (s.model == 0)
                        impl_->qwen->PrepareAlignment();
                    else
                        impl_->LoadAligner(requested_aligner, s.aligner_provider);
                    preparing(0, "Ready");
                }
                catch (...)
                {
                    preparing(0, "Unavailable");
                    throw;
                }
            },
            [&]
            {
                if (s.diarizer_directory.empty())
                    return;
                preparing(1, "Loading model");
                try
                {
                    impl_->LoadDiarizer(s);
                    preparing(1, "Ready");
                }
                catch (...)
                {
                    preparing(1, "Unavailable");
                    throw;
                }
            });
    }
    // Only time spent waiting on preparation is excluded from processing RTF.
    // Loading that overlaps inference must not be subtracted a second time.
    auto await_model = [&](std::future<void>& ready, int step)
    {
        if (!ready.valid())
            return;
        const auto begin = std::chrono::steady_clock::now();
        ready.wait();
        optional_setup += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        preparing(step, "");
        ready.get();
    };
    auto await_alignment = [&]
    {
        if (requested_aligner.empty())
            return;
        await_model(loading->alignment, 0);
        impl_->alignment_progress = [&](const auto& event)
        {
            auto report = event;
            if (report.stage == din::common::ProgressStage::Transcribing)
                report.stage = din::common::ProgressStage::Aligning;
            report.detail = "Word alignment: " + event.detail;
            if (progress)
                progress(report);
        };
    };
    if (progress)
        progress({din::common::ProgressStage::Transcribing, "Processing audio", 0, audio.Duration()});
    std::string language = s.language == "auto" ? "en" : s.language;
    std::vector<Timing> alignment_spans;
    // RNNT spans use the model's token/audio mapping and tokenizer, not cleaned
    // token labels (which lose subword boundaries). Recognition runs only once.
    auto token_spans = [&](const std::vector<int64_t>& tokens, const auto& starts)
    {
        if (tokens.empty())
            return;
        if (audio.Duration() <= 180)
        {
            alignment_spans.push_back({out.text, 0, audio.Duration()});
            return;
        }
        size_t begin = 0;
        while (begin < tokens.size())
        {
            const double from = std::max(0., starts.at(begin) * impl_->frame_rate - .1);
            size_t end = begin + 1;
            while (end < tokens.size() && starts.at(end) * impl_->frame_rate - from < 120)
                ++end;
            const double to = end < tokens.size() ? starts.at(end) * impl_->frame_rate : audio.Duration();
            std::vector<int64_t> part(tokens.begin() + begin, tokens.begin() + end);
            alignment_spans.push_back({impl_->tokenizer->Decode(part), from, to});
            begin = end;
        }
    };
    if (s.model == 0)
    {
        asr::qwen3::TranscriptionResult r;
        try
        {
            r = impl_->qwen->Transcribe(audio, await_alignment);
        }
        catch (const std::exception& e)
        {
            if (requested_aligner.empty() || !impl_->qwen_alignment)
                throw;
            // Even an ASR failure can arrive while alignment is still loading.
            // Join before replacing the Qwen instance used by the loader.
            const auto retry_setup = std::chrono::steady_clock::now();
            loading->thread.join();
            out.warning = std::string("Qwen alignment unavailable; retranscribed without alignment. ") + e.what();
            impl_->Load(s, false);
            optional_setup += std::chrono::duration<double>(std::chrono::steady_clock::now() - retry_setup).count();
            r = impl_->qwen->Transcribe(audio);
        }
        if (!r.reached_eos)
        {
            if (!out.warning.empty())
                out.warning += " ";
            out.warning += "Qwen transcription is incomplete: a chunk reached the token limit. Increase Max tokens "
                           "and transcribe again.";
        }
        out.text = std::move(r.text);
        out.timing_kind = "word";
        for (auto& t : r.timestamps)
            out.timings.push_back({std::move(t.text), t.start_time, t.end_time});
    }
    else if (s.model == 1)
    {
        auto r = impl_->whisper->Transcribe(audio);
        out.text = std::move(r.text);
        language = std::move(r.language);
        out.timing_kind = "segment";
        for (auto& segment : r.segments)
            out.timings.push_back({std::move(segment.text), segment.start, segment.end});
        alignment_spans = out.timings;
    }
    else if (s.model == 2)
    {
        auto r = impl_->parakeet->Transcribe(audio);
        out.text = std::move(r.text);
        out.timing_kind = "token";
        for (size_t i = 0; i < r.generated.tokens.size(); ++i)
            out.timings.push_back({impl_->tokenizer->CleanToken(r.generated.tokens[i]),
                                   r.generated.starts.at(i) * impl_->frame_rate,
                                   (r.generated.starts.at(i) + r.generated.durations.at(i)) * impl_->frame_rate});
        if (!requested_aligner.empty())
            token_spans(r.generated.tokens, r.generated.starts);
    }
    else
    {
        auto r = impl_->nemotron->Transcribe(audio);
        out.text = std::move(r.text);
        out.timing_kind = "token";
        for (size_t i = 0; i < r.generated.tokens.size(); ++i)
            out.timings.push_back({impl_->tokenizer->CleanToken(r.generated.tokens[i]),
                                   r.generated.starts.at(i) * impl_->frame_rate, std::nullopt});
        if (!requested_aligner.empty())
            token_spans(r.generated.tokens, r.generated.starts);
    }
    out.audio_seconds = s.model <= 1 ? completed_audio : audio.Duration();
    if (s.model != 0 && !requested_aligner.empty())
    {
        // Show alignment as a separate stage; final RTF includes its execution time.
        AlignTranscript(out, audio, alignment_spans,
                        [&](const Audio& slice, const std::string& text)
                        {
                            await_alignment();
                            std::vector<Timing> words;
                            for (auto& t : impl_->aligner->Align(slice, text, language))
                                words.push_back({std::move(t.text), t.start_time, t.end_time});
                            return words;
                        });
    }
    if (!s.diarizer_directory.empty())
    {
        try
        {
            await_model(loading->diarization, 1);
            impl_->diarization_progress = progress;
            const auto speakers = impl_->diarizer->Diarize(audio);
            for (const auto& segment : speakers.segments)
            {
                out.speaker_activity.push_back({"", segment.start_time, segment.end_time, segment.speaker});
                out.speakers.try_emplace(segment.speaker, "Speaker " + std::to_string(segment.speaker + 1));
            }
            for (auto& timing : out.timings)
                timing.speaker = speakers.SpeakerAt(static_cast<float>(timing.start),
                                                    static_cast<float>(timing.end.value_or(timing.start)));
            BuildSpeakerTurns(out);
        }
        catch (const std::exception& e)
        {
            if (!out.warning.empty())
                out.warning += " ";
            out.warning += std::string("Diarization unavailable; transcript and timing retained. ") + e.what();
        }
    }
    if (out.timings.empty())
        out.timing_kind.clear();
    // Silent audio can skip alignment entirely; still finish preparation before
    // reporting completion or allowing callbacks captured by this job to expire.
    if (loading && loading->alignment.valid())
    {
        try
        {
            await_model(loading->alignment, 0);
        }
        catch (const std::exception& e)
        {
            if (!out.warning.empty())
                out.warning += " ";
            out.warning += std::string("Alignment preparation unavailable. ") + e.what();
        }
    }
    out.setup_seconds += optional_setup;
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() - optional_setup;
    if (progress)
        progress({din::common::ProgressStage::Transcribing, "Complete", out.audio_seconds, audio.Duration()});
    return out;
#else
    (void)progress;
    (void)preparation;
    throw std::runtime_error("This is the standalone audio build. Build from the repository root with "
                             "DIN_BUILD_AUDIO_UI=ON for native ASR.");
#endif
}
Worker::Worker()
    : thread_(
          [this]
          {
              Loop();
          })
{
}
Worker::~Worker()
{
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
        jobs_.clear();
    }
    wake_.notify_one();
    thread_.join();
}
void Worker::Submit(Job job)
{
    {
        std::lock_guard lock(mutex_);
        jobs_.push_back(std::move(job));
    }
    wake_.notify_one();
}
void Worker::Import(size_t index, std::filesystem::path path)
{
    Submit({index, std::move(path), {}, {}});
}
void Worker::Transcribe(size_t index, std::shared_ptr<const Audio> audio, Settings settings)
{
    Submit({index, {}, std::move(audio), std::move(settings)});
}
void Worker::Loop()
{
    for (;;)
    {
        Job job;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock,
                       [this]
                       {
                           return stop_ || !jobs_.empty();
                       });
            if (stop_)
                return;
            job = std::move(jobs_.front());
            jobs_.pop_front();
            busy.store(true);
        }
        Completion completion;
        completion.index = job.index;
        completion.importing = !job.audio;
        try
        {
            if (completion.importing)
            {
                Report(job.index, {din::common::ProgressStage::DecodingAudio, "Decoding audio"});
                completion.audio = Decode(job.path);
                completion.peaks = Peaks(*completion.audio);
            }
            else
                completion.result = inference_.Run(
                    job.settings, *job.audio,
                    [this, index = job.index](const auto& p)
                    {
                        Report(index, p);
                    },
                    [this, index = job.index](int step, const std::string& detail)
                    {
                        std::lock_guard lock(mutex_);
                        if (progress_ && progress_->index == index)
                            progress_->preparation.at(step) = detail;
                    });
        }
        catch (const std::exception& e)
        {
            completion.error = e.what();
        }
        {
            std::lock_guard lock(mutex_);
            completed_.push_back(std::move(completion));
            progress_.reset();
            busy.store(false);
        }
    }
}
std::vector<Completion> Worker::Poll()
{
    std::lock_guard lock(mutex_);
    std::vector<Completion> result;
    result.swap(completed_);
    return result;
}
void Worker::Report(size_t index, const din::common::InferenceProgress& event)
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    if (!progress_ || progress_->index != index)
    {
        progress_ = JobProgress{};
        progress_->index = index;
        progress_->stage_started = now;
    }
    auto& p = *progress_;
    if (p.event.stage != event.stage || p.event.detail != event.detail)
        p.stage_started = now;
    if (event.stage == din::common::ProgressStage::Transcribing)
    {
        if (p.inference_started == std::chrono::steady_clock::time_point{})
            p.inference_started = now;
        if (event.completed_audio_seconds > p.event.completed_audio_seconds)
            p.measured_seconds = std::chrono::duration<double>(now - p.inference_started).count();
    }
    p.event = event;
}
std::optional<JobProgress> Worker::Progress()
{
    std::lock_guard lock(mutex_);
    return progress_;
}
}  // namespace din::studio
