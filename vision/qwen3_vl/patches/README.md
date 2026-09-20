# Optional TensorRT RTX provider build

The Python runtime works with the installed provider. The optional build below
adds persistent decoder workspace and preserves CUDA captures across shape
changes. It does not add an ONNX operator or change the exported model.

Tested source: [NVIDIA/TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI),
commit `478e52121167d0390cf7791dd10794bb0b7feabb` (Apache-2.0).
Persistent workspace is already in that source revision. Our accompanying patch
adds a bounded cache of execution contexts to the EPContext loading path. All
contexts share the same `ICudaEngine` and workspace. The cache requires an external
stream, CUDA graphs and `nv_persistent_context_memory=1`; this runtime enables it
only for the decoder when `--ep-library` is supplied. Calls must remain serial.

The cache retains up to 16 input-shape combinations. Further shapes reuse the
active context normally and may recapture. Output address changes are still
checked by the provider's binding code. Runtime kernel caches are stored per
shape; these are not additional serialized engines or model weight banks.

From the repository root, in a VS 2022 build environment with CUDA 13 and CMake:

```powershell
git clone https://github.com/NVIDIA/TensorRT-RTX-EP-ABI.git artifacts/trt-ep-source/repo
git -C artifacts/trt-ep-source/repo checkout 478e52121167d0390cf7791dd10794bb0b7feabb
$patch = (Resolve-Path vision/qwen3_vl/patches/trt-rtx-shared-contexts.patch).Path
git -C artifacts/trt-ep-source/repo apply --check $patch
git -C artifacts/trt-ep-source/repo apply $patch
cmake -S artifacts/trt-ep-source/repo -B artifacts/trt-ep-source/build -G 'Visual Studio 17 2022' -A x64 -DONNXRUNTIME_ROOT=D:/din-deploy/build-qwen3/_deps/onnxruntime/onnxruntime-win-x64-1.27.0 -DTRT_RTX_ROOT=D:/TensorRT-RTX-1.6.1.120 -DBUILD_TESTS=OFF -DUSE_PRECOMPILED_HOST_PROTOC=ON
cmake --build artifacts/trt-ep-source/build --config Release --parallel 4 -- /p:SpectreMitigation=false
$dependencies = '.venv/Lib/site-packages/onnxruntime_ep_nv_tensorrt_rtx'
foreach ($name in @('tensorrt_rtx_1_6.dll','tensorrt_onnxparser_rtx_1_6.dll','tensorrt_plugins.dll','cudart64_13.dll')) {
    Copy-Item -LiteralPath (Join-Path $dependencies $name) -Destination artifacts/trt-ep-source/build/Release
}
```

Adjust the two SDK paths for another machine. The local benchmark build used the
normal MSVC runtime because optional Spectre CRT libraries were absent. No installed
provider is replaced. Pass the built DLL explicitly:

```powershell
python -X utf8 -m vision.qwen3_vl.cli --model-dir artifacts/qwen3-vl-2b-bf16-prefix --ep-library artifacts/trt-ep-source/build/Release/onnxruntime_providers_nv_tensorrt_rtx.dll --images test.jpg --repeat 3
```

The provider library digest is part of the engine-cache identity. Rebuilding the
DLL therefore creates a fresh compatible cache on first use. This patch is local
and has not been submitted to or accepted by NVIDIA.
