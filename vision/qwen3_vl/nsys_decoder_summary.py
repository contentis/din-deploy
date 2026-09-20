# SPDX-License-Identifier: Apache-2.0
"""Summarize decoder-only CUDA graph work from an Nsight Systems SQLite export.

Projection labels assume this Qwen3-VL export's four fused matmuls per layer:
QKV, attention output, gate/up, down; then one vocabulary projection.
"""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def union_duration(intervals):
    total, end = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def timeline_summary(connection, bounds=None):
    """GPU activity across the captured range, including gaps between launches.

    The range runs from the first to the last recorded GPU kernel. This is a
    process CUDA timeline measure, not SM occupancy or a device-wide idle counter.
    Use a decoder-only NVTX capture with no concurrent inference for comparisons.
    """
    kernels = connection.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    begin, end = bounds or (min(x[0] for x in kernels), max(x[1] for x in kernels))
    kernels = [(max(begin, a), min(end, b)) for a, b in kernels if a < end and b > begin]
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    activity = list(kernels)
    transfers = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            activity.extend(
                (max(begin, start), min(end, stop))
                for start, stop in connection.execute(
                    f"SELECT start,end FROM {table} WHERE start<? AND end>?", (end, begin)
                )
            )
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        transfers = [
            {"copy_kind": kind, "calls": count, "bytes": size, "largest_bytes": largest}
            for kind, count, size, largest in connection.execute(
                "SELECT copyKind,count(*),sum(bytes),max(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE start<? AND end>? GROUP BY copyKind",
                (end, begin),
            )
        ]
    apis = (
        [
            {"name": name, "calls": count, "cpu_duration_ms": duration / 1e6}
            for name, count, duration in connection.execute(
                "SELECT s.value,count(*),sum(r.end-r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON s.id=r.nameId WHERE r.start<? AND r.end>? "
                "GROUP BY r.nameId ORDER BY count(*) DESC",
                (end, begin),
            )
        ]
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables
        else []
    )
    span = end - begin
    idle = span - union_duration(activity)
    return {
        "span_ms": span / 1e6,
        "kernel_idle_percent": 100 * (span - union_duration(kernels)) / span,
        "kernel_and_copy_idle_percent": 100 * idle / span,
        "kernel_and_copy_idle_ms": idle / 1e6,
        "transfers": transfers,
        "runtime_apis": apis,
        "scope": "First to last selected graph kernel" if bounds else "First to last captured GPU kernel",
    }


def summarize(path, metadata, graph_id=None):
    config = metadata["text_config"]
    connection = sqlite3.connect(path)
    if graph_id is None:
        graph_id = connection.execute(
            "SELECT graphId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL "
            "GROUP BY graphId ORDER BY count(DISTINCT correlationId) DESC LIMIT 1"
        ).fetchone()[0]
    records = connection.execute(
        "SELECT k.correlationId,k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.demangledName WHERE k.graphId=? ORDER BY k.start",
        (graph_id,),
    ).fetchall()
    invocations, groups = defaultdict(list), defaultdict(list)
    for correlation, start, end, name in records:
        invocations[correlation].append((start, end, name))
        groups[name].append(end - start)
    categories = defaultdict(list)
    projection_names = ["qkv", "attention_output", "gate_up", "down"]
    spans, unions, starts, ends = [], [], [], []
    for records in invocations.values():
        matrices = [r for r in records if "cudnn_generated_matMul" in r[2]]
        expected = 4 * config["num_hidden_layers"] + 1
        if len(matrices) != expected:
            raise ValueError(f"Expected {expected} decoder projections, found {len(matrices)}; inspect graph ID/export")
        for i, (start, end, _) in enumerate(matrices):
            name = "lm_head" if i == len(matrices) - 1 else projection_names[i % 4]
            categories[name].append(end - start)
        start, end = min(r[0] for r in records), max(r[1] for r in records)
        starts.append(start)
        ends.append(end)
        spans.append(end - start)
        unions.append(union_duration([(r[0], r[1]) for r in records]))
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    element_size = 4 if metadata["dtype"] == "float32" else 2
    weight_bytes = {
        "qkv": hidden * (hidden + 2 * kv) * element_size,
        "attention_output": hidden * hidden * element_size,
        "gate_up": hidden * intermediate * 2 * element_size,
        "down": hidden * intermediate * element_size,
        "lm_head": hidden * config["vocab_size"] * element_size,
    }
    count = len(invocations)
    projections = {
        name: {
            "calls_per_token": len(durations) / count,
            "kernel_sum_ms_per_token": sum(durations) / count / 1e6,
            "mean_us_per_call": statistics.mean(durations) / 1e3,
            "weight_bytes_per_call": weight_bytes[name],
            "inferred_weight_GB_per_second": weight_bytes[name] / (statistics.mean(durations) / 1e9) / 1e9,
        }
        for name, durations in categories.items()
    }
    return {
        "source": str(path),
        "capture_timeline": timeline_summary(connection),
        "graph_timeline": timeline_summary(connection, (min(starts), max(ends))),
        "graph_id": graph_id,
        "tokens": count,
        "kernels_per_token": len(records) if count == 1 else sum(len(r) for r in invocations.values()) / count,
        "mean_graph_span_ms": statistics.mean(spans) / 1e6,
        "mean_kernel_union_ms": statistics.mean(unions) / 1e6,
        "mean_sum_kernel_ms": sum(sum(d) for d in groups.values()) / count / 1e6,
        "mean_between_graph_gap_ms": statistics.mean([s - e for s, e in zip(starts[1:], ends[:-1], strict=True)]) / 1e6
        if count > 1
        else None,
        "projections": projections,
        "kernel_groups": [
            {"name": name, "calls_per_token": len(times) / count, "sum_ms_per_token": sum(times) / count / 1e6}
            for name, times in sorted(groups.items(), key=lambda item: sum(item[1]), reverse=True)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--graph-id", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.sqlite, json.loads(args.metadata.read_text(encoding="utf-8")), args.graph_id)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "kernel_groups"}, indent=2))


if __name__ == "__main__":
    main()
