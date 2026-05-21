#!/usr/bin/env python3
"""
Batch-size scaling analyzer for the Tiny LLaMA V1 baseline.

Reads `performance_summary.json` produced by `tiny_llama_v1.py` from one or
more profile directories (typically `scaling_bs*/`) and prints a
batch-size-vs-throughput-vs-memory comparison table. It also computes
per-batch derived metrics (linear-scaling efficiency, memory per sample)
and writes a machine-readable summary to disk.

Usage:
    python analyze_batch_scaling.py --profile-dirs scaling_bs*
    python analyze_batch_scaling.py --profile-dirs scaling_bs4 scaling_bs8 scaling_bs16
    python analyze_batch_scaling.py --profile-dirs 'scaling_bs*' \\
        --output-json batch_scaling_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class BatchPoint:
    profile_dir: Path
    batch_size: int
    seq_len: int
    num_steps: int
    avg_batch_time_s: float
    avg_forward_time_s: float
    avg_backward_time_s: float
    avg_optimizer_time_s: float
    avg_training_speed_sps: float  # samples/sec, as reported
    avg_loss: float
    peak_memory_mb: Optional[float]
    avg_peak_memory_mb: Optional[float]
    avg_memory_mb: Optional[float]
    gpu_name: str
    pytorch_version: str

    @property
    def tokens_per_sec(self) -> float:
        return self.avg_training_speed_sps * self.seq_len

    @property
    def memory_per_sample_mb(self) -> Optional[float]:
        peak = self.peak_memory_mb if self.peak_memory_mb is not None else self.avg_peak_memory_mb
        if peak is None or self.batch_size <= 0:
            return None
        return peak / self.batch_size

    @property
    def time_per_sample_ms(self) -> float:
        if self.batch_size <= 0:
            return 0.0
        return (self.avg_batch_time_s * 1000.0) / self.batch_size


def _expand_profile_dirs(patterns: List[str]) -> List[Path]:
    """Expand shell-style globs and bare paths into a deduplicated, sorted list of dirs."""
    seen: List[Path] = []
    for p in patterns:
        matches = sorted(glob.glob(p)) or [p]
        for m in matches:
            path = Path(m)
            if path.is_dir() and path not in seen:
                seen.append(path)
    return seen


def _load_point(profile_dir: Path) -> Optional[BatchPoint]:
    summary_path = profile_dir / "performance_summary.json"
    if not summary_path.exists():
        print(f"  [skip] {profile_dir}: no performance_summary.json", file=sys.stderr)
        return None
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  [skip] {profile_dir}: invalid JSON ({e})", file=sys.stderr)
        return None

    perf = data.get("performance_summary", {}) or {}
    cfg = data.get("config", {}) or {}
    train = data.get("training_params", {}) or {}
    sysinfo = data.get("system_info", {}) or {}

    # Tiny LLaMA's monitor reports times in seconds and speed in samples/sec.
    return BatchPoint(
        profile_dir=profile_dir,
        batch_size=int(train.get("batch_size", 0)),
        seq_len=int(cfg.get("max_seq_len", 0)),
        num_steps=int(train.get("num_steps", 0)),
        avg_batch_time_s=float(perf.get("avg_batch_time", 0.0)),
        avg_forward_time_s=float(perf.get("avg_forward_time", 0.0)),
        avg_backward_time_s=float(perf.get("avg_backward_time", 0.0)),
        avg_optimizer_time_s=float(perf.get("avg_optimizer_time", 0.0)),
        avg_training_speed_sps=float(perf.get("avg_training_speed", 0.0)),
        avg_loss=float(perf.get("avg_loss", 0.0)),
        peak_memory_mb=(
            float(perf["peak_memory_mb"]) if "peak_memory_mb" in perf else None
        ),
        avg_peak_memory_mb=(
            float(perf["avg_peak_memory_mb"])
            if "avg_peak_memory_mb" in perf else None
        ),
        avg_memory_mb=(
            float(perf["avg_memory_mb"]) if "avg_memory_mb" in perf else None
        ),
        gpu_name=str(sysinfo.get("gpu_name", "unknown")),
        pytorch_version=str(sysinfo.get("pytorch_version", "unknown")),
    )


def _print_table(points: List[BatchPoint]) -> None:
    # Header
    cols = [
        ("BS",        4,  "{:>4d}"),
        ("seq",       4,  "{:>4d}"),
        ("step ms",   8,  "{:>8.2f}"),
        ("fwd ms",    7,  "{:>7.2f}"),
        ("bwd ms",    7,  "{:>7.2f}"),
        ("opt ms",    7,  "{:>7.2f}"),
        ("samp/s",    8,  "{:>8.1f}"),
        ("tok/s",     9,  "{:>9.0f}"),
        ("ms/samp",   8,  "{:>8.2f}"),
        ("peak MB",   9,  "{:>9.1f}"),
        ("MB/samp",   9,  "{:>9.2f}"),
        ("scale eff", 9,  "{:>8.1f}%"),
        ("loss",      7,  "{:>7.4f}"),
    ]
    header = "  ".join(f"{name:>{w}}" for name, w, _ in cols)
    print(header)
    print("  ".join("-" * w for _, w, _ in cols))

    # Reference for "scale eff": linear scaling vs. the smallest batch size.
    base = min(points, key=lambda p: p.batch_size)
    base_sps_per_sample = (
        base.avg_training_speed_sps / base.batch_size if base.batch_size else 0.0
    )

    for p in sorted(points, key=lambda x: x.batch_size):
        peak = p.peak_memory_mb if p.peak_memory_mb is not None else (
            p.avg_peak_memory_mb if p.avg_peak_memory_mb is not None else float("nan")
        )
        mb_per_sample = p.memory_per_sample_mb if p.memory_per_sample_mb is not None else float("nan")
        if base_sps_per_sample > 0 and p.batch_size > 0:
            this_per_sample = p.avg_training_speed_sps / p.batch_size
            # Higher samples/sec/sample than base means superlinear scaling.
            scale_eff = (this_per_sample / base_sps_per_sample) * 100.0
        else:
            scale_eff = float("nan")

        row = [
            ("BS",        p.batch_size),
            ("seq",       p.seq_len),
            ("step ms",   p.avg_batch_time_s * 1000.0),
            ("fwd ms",    p.avg_forward_time_s * 1000.0),
            ("bwd ms",    p.avg_backward_time_s * 1000.0),
            ("opt ms",    p.avg_optimizer_time_s * 1000.0),
            ("samp/s",    p.avg_training_speed_sps),
            ("tok/s",     p.tokens_per_sec),
            ("ms/samp",   p.time_per_sample_ms),
            ("peak MB",   peak),
            ("MB/samp",   mb_per_sample),
            ("scale eff", scale_eff),
            ("loss",      p.avg_loss),
        ]
        print("  ".join(fmt.format(v) for (_, _, fmt), (_, v) in zip(cols, row)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dirs",
        nargs="+",
        required=True,
        help="Profile directories (or glob patterns like 'scaling_bs*').",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write a machine-readable summary JSON.",
    )
    args = parser.parse_args()

    dirs = _expand_profile_dirs(args.profile_dirs)
    if not dirs:
        print("No matching profile directories found.", file=sys.stderr)
        return 1

    print(f"Found {len(dirs)} profile dir(s):")
    for d in dirs:
        print(f"   {d}")
    print()

    points = [p for p in (_load_point(d) for d in dirs) if p is not None]
    if not points:
        print("No usable performance_summary.json files found.", file=sys.stderr)
        return 1

    # Header line with environment context.
    gpus = {p.gpu_name for p in points}
    pts = {p.pytorch_version for p in points}
    if len(gpus) == 1:
        print(f"GPU: {next(iter(gpus))}")
    else:
        print(f"GPUs (mixed!): {sorted(gpus)}")
    if len(pts) == 1:
        print(f"PyTorch: {next(iter(pts))}")
    print()

    _print_table(points)

    if args.output_json:
        out = {
            "points": [
                {
                    "profile_dir": str(p.profile_dir),
                    "batch_size": p.batch_size,
                    "seq_len": p.seq_len,
                    "num_steps": p.num_steps,
                    "avg_batch_time_s": p.avg_batch_time_s,
                    "avg_forward_time_s": p.avg_forward_time_s,
                    "avg_backward_time_s": p.avg_backward_time_s,
                    "avg_optimizer_time_s": p.avg_optimizer_time_s,
                    "avg_training_speed_sps": p.avg_training_speed_sps,
                    "tokens_per_sec": p.tokens_per_sec,
                    "time_per_sample_ms": p.time_per_sample_ms,
                    "peak_memory_mb": p.peak_memory_mb,
                    "avg_peak_memory_mb": p.avg_peak_memory_mb,
                    "memory_per_sample_mb": p.memory_per_sample_mb,
                    "avg_loss": p.avg_loss,
                    "gpu_name": p.gpu_name,
                    "pytorch_version": p.pytorch_version,
                }
                for p in sorted(points, key=lambda x: x.batch_size)
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSummary JSON written to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
