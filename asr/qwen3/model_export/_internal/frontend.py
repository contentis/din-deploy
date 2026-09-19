# SPDX-License-Identifier: Apache-2.0
"""Whisper-style ONNX log-mel frontend and HF-generated native prompt assets."""

import json
import sys
from pathlib import Path

import torch
from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from common.model_export.log_mel import LogMel as WhisperMel  # noqa: E402


class LogMel(WhisperMel):
    def __init__(self, processor):
        fe = processor.feature_extractor
        if (fe.n_fft, fe.hop_length, fe.feature_size, fe.sampling_rate, fe.dither) != (400, 160, 128, 16000, 0):
            raise ValueError("Native frontend requires the standard Qwen3 16 kHz configuration")
        super().__init__(fe.mel_filters.T.copy(), torch.float32, frames=None)


def save_native_assets(processor, output, task):
    tokenizer = processor.tokenizer

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    data = {
        "audio_start": encode(processor.audio_bos_token),
        "audio_end": encode(processor.audio_eos_token),
    }
    if task == "asr":
        prefixes = {}
        suffixes = {}
        languages = {}
        for language in [None, *LANGUAGE_CODE_TO_NAME.values()]:
            messages = [{"role": "user", "content": [{"type": "audio"}]}]
            rendered = tokenizer.apply_chat_template(
                messages, chat_template=processor.chat_template, tokenize=False, add_generation_prompt=True
            )
            prefix, suffix = rendered.split(processor.audio_token)
            prefixes[language or "auto"] = encode(prefix)
            suffixes[language or "auto"] = encode(suffix + (f"language {language}<asr_text>" if language else ""))
            languages[language or "auto"] = language or ""
        for code, language in LANGUAGE_CODE_TO_NAME.items():
            prefixes[code] = prefixes[language]
            suffixes[code] = suffixes[language]
            languages[code] = language
        data["prefixes"] = prefixes
        data["suffixes"] = suffixes
        data["languages"] = languages
        data["suffix"] = suffixes["auto"]
    (output / "native.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
