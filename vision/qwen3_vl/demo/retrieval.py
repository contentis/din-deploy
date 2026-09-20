# SPDX-License-Identifier: Apache-2.0
"""Experimental image/text retrieval using Qwen3-VL hidden states and SQLite.

No database server or additional dependency: SQLite persists vectors and NumPy
performs exact cosine search. Instruct hidden states are not contrastively trained;
ranking quality must be evaluated before relying on this experiment.
"""

import argparse
import hashlib
import heapq
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

from ..inputs import prepare_inputs
from ..runtime import Qwen3VL

EMBED_PROMPT = "Summarize the content."


class VectorStore:
    def __init__(self, path, identity):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        existing = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if existing and existing[0] != identity:
            self.db.close()
            raise ValueError("Index model, precision or preprocessing differs; create a separate database")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (identity,))

    def close(self):
        self.db.close()

    @staticmethod
    def normalized(vector):
        vector = np.asarray(vector, dtype="<f4").reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Embedding must be finite and nonzero")
        return vector / norm

    def upsert(self, path, vector):
        vector = self.normalized(vector)
        with self.db:
            self.db.execute(
                "INSERT INTO images VALUES (?, ?) ON CONFLICT(path) DO UPDATE SET vector=excluded.vector",
                (str(Path(path).resolve()), vector.tobytes()),
            )

    def search(self, vector, k=5):
        if k < 1:
            raise ValueError("k must be positive")
        query = self.normalized(vector)
        best = []
        cursor = self.db.execute("SELECT path, vector FROM images ORDER BY path")
        while rows := cursor.fetchmany(128):
            matrix = np.stack([np.frombuffer(blob, dtype="<f4") for _, blob in rows])
            if matrix.shape[1] != query.size:
                raise ValueError("Index vector dimension differs from query")
            for (path, _), score in zip(rows, matrix @ query, strict=True):
                item = (float(score), path)
                if len(best) < k:
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)
        return [{"path": path, "cosine": score} for score, path in sorted(best, reverse=True)]


@torch.inference_mode()
def embed(runner, budget, image=None, text=None):
    prompt = f"{text}\n{EMBED_PROMPT}" if text else EMBED_PROMPT
    inputs, _ = prepare_inputs(
        runner.processor,
        prompt,
        images=[image] if image else [],
        max_visual_tokens=budget,
        max_image_tokens=max(runner.metadata["vision_buckets"]) // 4,
    )
    result = runner.prefill(inputs)
    runner.synchronize()
    return result["hidden"][0].float().cpu().numpy().copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"], default="trt-rtx")
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("images", type=Path, nargs="+")
    search = sub.add_parser("search")
    query = search.add_mutually_exclusive_group(required=True)
    query.add_argument("--image", type=Path)
    query.add_argument("--text")
    search.add_argument("--k", type=int, default=3)
    search.add_argument("--reason", help="Question to answer by loading the retrieved images into Qwen3-VL")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    runner = Qwen3VL(args.model_dir, args.provider, threads=args.threads)
    identity = hashlib.sha256(
        json.dumps(
            {
                "graphs": runner.metadata["graphs"],
                "budget": args.max_visual_tokens,
                "method": "last_decoder_hidden_v1",
                "prompt": EMBED_PROMPT,
                "provider": args.provider,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = VectorStore(args.db, identity)
    try:
        if args.command == "ingest":
            for path in args.images:
                store.upsert(path, embed(runner, args.max_visual_tokens, image=path))
                print(f"Indexed {path}", flush=True)
        else:
            results = store.search(embed(runner, args.max_visual_tokens, args.image, args.text), args.k)
            print(json.dumps(results, indent=2))
            if args.reason and results:
                inputs, _ = prepare_inputs(
                    runner.processor,
                    args.reason,
                    images=[r["path"] for r in results],
                    max_visual_tokens=args.max_visual_tokens,
                    max_image_tokens=max(runner.metadata["vision_buckets"]) // 4,
                )
                print(json.dumps(runner.generate(inputs), ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
