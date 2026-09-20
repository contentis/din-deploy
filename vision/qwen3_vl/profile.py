# SPDX-License-Identifier: Apache-2.0
"""Summarize ORT partitioning; fail if inference crosses providers/partitions."""

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [event for event in events if event.get("cat") == "Node" and "provider" in event.get("args", {})]
    providers = Counter(event["args"]["provider"] for event in nodes)
    partitions = sorted({event["name"] for event in nodes})
    return {
        "path": str(path),
        "provider_calls": dict(providers),
        "partitions": partitions,
        "one_rtx_partition": len(partitions) == 1 and set(providers) == {"nv_tensorrt_rtx"},
        "execution_ms": sum(event.get("dur", 0) for event in nodes) / 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, nargs="+")
    parser.add_argument("--require-continuous", action="store_true")
    args = parser.parse_args()
    result = [summarize(path) for path in args.profiles]
    print(json.dumps(result, indent=2))
    if args.require_continuous and not all(r["one_rtx_partition"] for r in result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
