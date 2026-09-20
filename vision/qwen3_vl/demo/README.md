# Small retrieval experiment

This demo stores normalized Qwen3-VL final prompt hidden states in SQLite and uses
NumPy for exact cosine search. There is no additional database dependency or server.
It accepts image and text queries and can pass the retrieved images back to the
ONNX model to answer a question. Database metadata prevents mixing model exports,
providers or visual budgets. Re-ingesting a path replaces its vector.

```powershell
python -m vision.qwen3_vl.demo.retrieval --model-dir artifacts/qwen3-vl-2b-bf16 --db artifacts/images.sqlite ingest first.jpg second.jpg third.jpg
python -m vision.qwen3_vl.demo.retrieval --model-dir artifacts/qwen3-vl-2b-bf16 --db artifacts/images.sqlite search --image first.jpg --k 3
python -m vision.qwen3_vl.demo.retrieval --model-dir artifacts/qwen3-vl-2b-bf16 --db artifacts/images.sqlite search --text "a dog outdoors" --k 3 --reason "Describe the dogs in these images."
```

These are **experimental Instruct-model features**, not calibrated retrieval
embeddings. Self-retrieval checks the plumbing; it does not establish useful text
ranking or general retrieval accuracy. The next accuracy-oriented iteration should
use [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), which is
trained for multimodal retrieval and has its own input/pooling contract. Do not
silently substitute that checkpoint into the Instruct exporter.

For a larger index, [sqlite-vec](https://github.com/asg017/sqlite-vec) is a small,
pure-C SQLite extension with vector tables and Python bindings. This first demo
uses the existing NumPy dependency to avoid native extension installation and to
keep the index inspectable. Search is O(number of images × embedding dimension)
with bounded batches of 128 vectors; it is intended for small local collections.
