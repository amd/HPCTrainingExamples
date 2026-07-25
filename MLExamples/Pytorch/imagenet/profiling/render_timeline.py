#!/usr/bin/env python3
"""Render a compute-vs-communication timeline PNG from Chrome/Perfetto JSON traces.

Fully headless (matplotlib 'Agg' backend) -- no X server, browser, or Java needed.
Each input JSON is one rank; every rank becomes one lane. GPU kernels are colored
by class: RCCL/NCCL collectives (communication) vs everything else (compute).

Usage:
  render_timeline.py --out ../figs/timeline_4gpu.png rank0.json rank1.json ...
  render_timeline.py --out out.png --glob 'torch_trace_rank*.json'
"""
import argparse
import bisect
import glob as _glob
import gzip
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# A GPU kernel is "communication" if its name matches these RCCL/NCCL patterns.
COMM_RE = re.compile(r"nccl|rccl|all[_ ]?reduce|allgather|all[_ ]?gather|"
                     r"reduce[_ ]?scatter|broadcast|ncclDevKernel|ncclKernel",
                     re.IGNORECASE)


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        data = json.load(fh)
    # Chrome trace can be a dict with "traceEvents" or a bare list.
    return data["traceEvents"] if isinstance(data, dict) else data


def _gpu_kernels(events):
    """Return [(start_us, dur_us, name, is_comm)] for device-side GPU kernels.

    Only true device kernels (Kineto cat == "kernel") count as GPU-busy time.
    We deliberately exclude the `nccl:all_reduce` *user annotations* (cat
    "gpu_user_annotation"), which span the whole collective incl. waiting and
    would massively overstate communication; the real GPU comm is the
    `ncclDevKernel*` kernel itself.
    """
    out = []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e or "ts" not in e:
            continue
        if "kernel" not in str(e.get("cat", "")).lower():
            continue
        name = str(e.get("name", ""))
        out.append((float(e["ts"]), float(e["dur"]), name, bool(COMM_RE.search(name))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", help="one Chrome/Perfetto JSON per rank")
    ap.add_argument("--glob", help="glob pattern for trace files (sorted)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default=None, help="figure title")
    ap.add_argument("--start-ms", type=float, default=None, help="crop window start (ms)")
    ap.add_argument("--end-ms", type=float, default=None, help="crop window end (ms)")
    ap.add_argument("--auto-window-ms", type=float, default=350.0,
                    help="if no crop given, show this many ms after the warmup skip")
    ap.add_argument("--warmup-frac", type=float, default=0.25,
                    help="skip this fraction of the trace as warmup before auto-window")
    args = ap.parse_args(argv)

    files = list(args.traces)
    if args.glob:
        files += sorted(_glob.glob(args.glob))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("render_timeline: no trace files found", file=sys.stderr)
        return 2
    files = sorted(dict.fromkeys(files))  # de-dup, keep order

    lanes = []          # list of (label, kernels)
    global_min = None
    for i, path in enumerate(files):
        try:
            ev = _load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"render_timeline: skip {path}: {exc}", file=sys.stderr)
            continue
        ks = _gpu_kernels(ev)
        if not ks:
            print(f"render_timeline: no GPU kernels in {path}", file=sys.stderr)
        m = re.search(r"rank(\d+)", os.path.basename(path))
        label = f"GPU {m.group(1)}" if m else f"GPU {i}"
        lanes.append((label, ks))
        for ts, _dur, _n, _c in ks:
            global_min = ts if global_min is None else min(global_min, ts)

    if global_min is None:
        print("render_timeline: no GPU kernels found in any trace", file=sys.stderr)
        return 3

    # Convert to ms relative to the global start; collect comm start times.
    all_end = 0.0
    comm_times = []
    for _, ks in lanes:
        for ts, dur, _n, is_comm in ks:
            x0 = (ts - global_min) / 1000.0
            all_end = max(all_end, x0 + dur / 1000.0)
            if is_comm:
                comm_times.append(x0)
    comm_times.sort()

    if args.start_ms is not None or args.end_ms is not None:
        w0 = args.start_ms if args.start_ms is not None else 0.0
        w1 = args.end_ms if args.end_ms is not None else all_end
    elif comm_times:
        # Center the window on the densest communication region so the
        # all-reduce is always visible.
        win = args.auto_window_ms
        best_start, best_count = comm_times[0], 0
        for t in comm_times:
            c = (bisect.bisect_right(comm_times, t + win)
                 - bisect.bisect_left(comm_times, t))
            if c > best_count:
                best_count, best_start = c, t
        w0 = max(0.0, best_start - win * 0.1)
        w1 = min(all_end, w0 + win)
    else:
        w0 = all_end * args.warmup_frac
        w1 = min(all_end, w0 + args.auto_window_ms)

    # Minimum drawn width so short comm kernels stay visible at this zoom.
    min_comm_w = (w1 - w0) * 0.0025

    fig, ax = plt.subplots(figsize=(14, 0.9 * len(lanes) + 2.2))
    lane_h = 0.62
    comm_bars_any = False
    for lane_idx, (label, ks) in enumerate(lanes):
        y = lane_idx
        comp, comm = [], []
        for ts, dur, _n, is_comm in ks:
            x0 = (ts - global_min) / 1000.0
            d = dur / 1000.0
            if x0 + d < w0 or x0 > w1:
                continue
            if is_comm:
                comm.append((x0, max(d, min_comm_w)))
            else:
                comp.append((x0, d))
        if comp:
            ax.broken_barh(comp, (y - lane_h / 2, lane_h),
                           facecolors="#2b8cbe", edgecolors="none")
        if comm:
            comm_bars_any = True
            ax.broken_barh(comm, (y - lane_h / 2, lane_h),
                           facecolors="#e6550d", edgecolors="none")

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([l for l, _ in lanes])
    ax.set_ylim(-0.7, len(lanes) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(w0, w1)
    ax.set_xlabel("time (ms)")
    title = args.title or (
        f"Measured GPU timeline: compute vs RCCL communication "
        f"({len(lanes)}x MI300A, DDP ResNet50)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    legend = [Patch(facecolor="#2b8cbe", label="compute (conv / GEMM / BN / etc.)"),
              Patch(facecolor="#e6550d", label="communication (RCCL all-reduce)")]
    ax.legend(handles=legend, loc="upper right", framealpha=0.95, fontsize=9)

    if not comm_bars_any:
        ax.text(0.01, 0.98,
                "note: no RCCL kernels landed in this window "
                "(on-package fabric is nearly free at 1 node)",
                transform=ax.transAxes, va="top", fontsize=8, color="#666666")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"render_timeline: wrote {args.out} "
          f"({len(lanes)} lanes, window {w0:.0f}-{w1:.0f} ms of {all_end:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
