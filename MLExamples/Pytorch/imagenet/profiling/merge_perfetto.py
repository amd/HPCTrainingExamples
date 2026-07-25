#!/usr/bin/env python3
"""Merge per-rank torch.profiler Chrome traces into one Perfetto-ingestible file.

Each input JSON is one rank (a valid Chrome/Perfetto Trace Event file). We remap
every event's pid to the rank index so Perfetto shows one process lane per GPU,
keep only the GPU-side slices (device kernels, GPU annotations, memcpy) to stay
small, and emit a single gzipped {"traceEvents":[...]} that loads directly in
https://ui.perfetto.org.

Usage:
  merge_perfetto.py --out traces/imagenet_4gpu.perfetto.json.gz \
      traces/torch_trace_rank0.json traces/torch_trace_rank1.json ...
"""
import argparse
import glob as _glob
import gzip
import json
import os
import re
import sys

KEEP_CATS = ("kernel", "gpu_user_annotation", "gpu_memcpy")


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        d = json.load(fh)
    return d["traceEvents"] if isinstance(d, dict) else d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", help="one torch.profiler Chrome JSON per rank")
    ap.add_argument("--glob", help="glob for the per-rank traces (sorted)")
    ap.add_argument("--out", required=True, help="output .json or .json.gz")
    args = ap.parse_args(argv)

    files = list(args.traces)
    if args.glob:
        files += sorted(_glob.glob(args.glob))
    files = sorted(dict.fromkeys(f for f in files if os.path.isfile(f)))
    if not files:
        print("merge_perfetto: no trace files", file=sys.stderr)
        return 2

    out = []
    for path in files:
        m = re.search(r"rank(\d+)", os.path.basename(path))
        rank = int(m.group(1)) if m else files.index(path)
        # process/thread names so the lane reads "GPU r".
        out.append({"ph": "M", "name": "process_name", "pid": rank, "tid": 0,
                    "args": {"name": f"GPU {rank}"}})
        out.append({"ph": "M", "name": "process_sort_index", "pid": rank, "tid": 0,
                    "args": {"sort_index": rank}})
        kept = 0
        for e in _load(path):
            if e.get("ph") != "X" or "dur" not in e or "ts" not in e:
                continue
            if str(e.get("cat", "")).lower() not in KEEP_CATS:
                continue
            out.append({"ph": "X", "cat": e.get("cat", ""), "name": e.get("name", ""),
                        "ts": e["ts"], "dur": e["dur"], "pid": rank,
                        "tid": e.get("tid", 0)})
            kept += 1
        print(f"merge_perfetto: {os.path.basename(path)} -> GPU {rank}: {kept} slices")

    doc = {"traceEvents": out, "displayTimeUnit": "ms"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wt") as fh:
        json.dump(doc, fh)
    print(f"merge_perfetto: wrote {args.out} ({len(out)} events, {len(files)} lanes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
