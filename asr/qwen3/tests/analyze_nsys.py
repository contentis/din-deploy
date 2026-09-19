# SPDX-License-Identifier: Apache-2.0
"""Internal CUDA/NVTX trace summary. Uses only the Python standard library."""

import argparse
import bisect
import json
import sqlite3
import statistics


def summarize(path):
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges = db.execute(
        "SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'qwen3.%' AND end IS NOT NULL ORDER BY start"
    ).fetchall()
    activity = []
    kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
    compute = kernel_table if kernel_table in tables else "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    for table in [compute, "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]:
        if table in tables:
            activity.extend(db.execute(f"SELECT start,end FROM {table}"))
    activity.sort()
    merged = []
    for start, end in activity:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])

    ends = [b for _, b in merged]
    prefix = [0]
    for a, b in merged:
        prefix.append(prefix[-1] + b - a)

    def integral(t):
        index = bisect.bisect_right(ends, t)
        return prefix[index] + (max(0, t - merged[index][0]) if index < len(merged) else 0)

    def busy(start, end):
        return integral(end) - integral(start)

    phases = {}
    for name in sorted({r[2] for r in ranges}):
        spans = [(a, b) for a, b, n in ranges if n == name]
        total = sum(b - a for a, b in spans)
        phases[name] = {
            "count": len(spans),
            "wall_ms": total / 1e6,
            "gpu_active_percent": 100 * sum(busy(a, b) for a, b in spans) / total,
        }
    # First iteration consumes prefill's token; final iteration may only consume EOS.
    steps = [(b - a) / 1e6 for a, b, n in ranges if n == "qwen3.decode_step"][1:-1]
    ordered = sorted(steps)
    latency = {
        "count": len(steps),
        "median_ms": statistics.median(steps),
        "p95_ms": ordered[int((len(ordered) - 1) * 0.95)],
        "p99_ms": ordered[int((len(ordered) - 1) * 0.99)],
        "first_32_median_ms": statistics.median(steps[:32]),
        "last_32_median_ms": statistics.median(steps[-32:]),
    }
    api = db.execute(
        "SELECT s.value,count(*),sum(r.end-r.start)/1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId GROUP BY s.value ORDER BY 3 DESC"
    ).fetchall()
    top_kernels = []
    if kernel_table in tables:
        top_kernels = db.execute(
            "SELECT s.value,count(*),sum(k.end-k.start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.shortName GROUP BY s.value ORDER BY 3 DESC LIMIT 15"
        ).fetchall()
    copies = db.execute(
        "SELECT copyKind,count(*),sum(bytes),sum(end-start)/1e6 "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
    ).fetchall()
    return {"phases": phases, "decode_latency": latency, "apis": api, "kernels": top_kernels, "copies": copies}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.sqlite)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({"phases": result["phases"], "decode_latency": result["decode_latency"]}, indent=2))
