# Nemotron diarization

[`nvidia/Nemotron-3-Diarization-preview`](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview): up to eight speakers, including overlaps, with 10 ms predictions.

| Feature | Model / upstream toolkit | C++ sample |
| --- | :---: | :---: |
| Offline / long form | ✅ | ✅ |
| Live streaming | ✅ | — |
| Batched recordings | ✅ | — |
| Standalone diarization | ✅ | ✅ |
| ASR + forced alignment + speaker labels | ✅ | ✅ |
| CPU (FP32) / GPU (TensorRT RTX) | | ✅ |

## Export and validate

Requires access to the gated model and NeMo Speech commit `f613eed86ed4696db0891aac4e9104337a39142c` (the PyPI 2.7.3 wheel is too old).

```bash
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git@f613eed86ed4696db0891aac4e9104337a39142c"
python asr/diarization/model_export/export_diarization.py --output artifacts/nemotron-diarization/onnx-bf16 --dtype bf16
python asr/diarization/model_export/validate_diarization.py --checkpoint model.nemo --model-dir artifacts/nemotron-diarization/onnx-bf16 --provider trt-rtx --audio audio-16khz.wav
```

HF downloads automatically after login. `--checkpoint` accepts an existing `.nemo` file. Export defaults to original FP32 weights; `--dtype fp16|bf16` enables reduced precision. NVIDIA documents BF16 inference; training precision is not specified. Validation runs the NeMo reference on CUDA.

## Run

```bash
cmake --build build --target din_diarization_cli
din_diarization_cli audio.wav --model-dir artifacts/nemotron-diarization/onnx-bf16
din_diarization_cli audio.wav --model-dir artifacts/nemotron-diarization/onnx-fp32 --provider cpu
din_diarization_cli audio.wav --asr-dir artifacts/qwen3/onnx-bf16 --aligner-dir artifacts/qwen3/aligner-onnx-bf16
```

The combined command runs diarization alongside Qwen3 ASR/alignment. Each component has its own provider selection. Other ASR models can use the independent `Pipeline` and `Result::SpeakerAt` with their word timestamps. Overlapping speech is retained in speaker intervals; ASR is not source separation.

One fixed-shape ONNX contains FP32 log-mel, model weights and speaker-cache updates. State stays on the selected device between chunks; recording length does not change the engine shape. Progress callbacks include diarization. FP32 validation checks probabilities strictly; reduced precision checks speaker/activity agreement against NeMo.

Model license: [NVIDIA evaluation license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/), internal evaluation on NVIDIA GPU systems only. Model artifacts and evaluation results must not be redistributed without permission.
