#!/usr/bin/env python3
"""Render a per-rank compute-vs-communication bar chart from a TAU pprof text dump.

Fully headless (matplotlib 'Agg') -- no Java, ParaProf, or X server needed.
Input is the output of `pprof -s` (see profiling/tau/tau_pprof.txt), which lists,
per NODE;CONTEXT;THREAD, the exclusive time of every GPU kernel. We aggregate the
GPU-device kernels ("[ROCm Kernel] ...") per rank (NODE) and split them into:
  * communication : RCCL/NCCL collectives (ncclDevKernel*)
  * compute       : every other GPU kernel (conv/gemm/batchnorm/elementwise/...)

Usage:
  render_tau_profile.py tau/tau_pprof.txt --out figs/tau_profile_4gpu.png
"""
import argparse
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NODE_RE = re.compile(r"^NODE (\d+);CONTEXT (\d+);THREAD (\d+):")
STOP_RE = re.compile(r"^FUNCTION SUMMARY")


def is_comm(name: str) -> bool:
    n = name.lower()
    return "nccl" in n or "rccl" in n


def parse(path: str):
    """Return {rank: {"compute": msec, "comm": msec}} from a pprof text dump."""
    ranks = {}
    cur = None
    in_summary = False
    with open(path) as fh:
        for line in fh:
            if STOP_RE.match(line):
                in_summary = True  # mean/total aggregates -> stop counting
            m = NODE_RE.match(line)
            if m:
                cur = int(m.group(1))
                ranks.setdefault(cur, {"compute": 0.0, "comm": 0.0})
                continue
            if in_summary or cur is None:
                continue
            if "[ROCm Kernel]" not in line:
                continue
            toks = line.split()
            if len(toks) < 7:
                continue
            try:
                excl = float(toks[1].replace(",", ""))
            except ValueError:
                continue
            name = line.split("[ROCm Kernel]", 1)[1]
            key = "comm" if is_comm(name) else "compute"
            ranks[cur][key] += excl
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pprof")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="TAU ParaProf: GPU compute vs RCCL communication per rank")
    args = ap.parse_args()

    ranks = parse(args.pprof)
    if not ranks:
        sys.exit("no per-rank GPU kernel data found in " + args.pprof)

    xs = sorted(ranks)
    compute = [ranks[r]["compute"] / 1000.0 for r in xs]  # -> seconds
    comm = [ranks[r]["comm"] / 1000.0 for r in xs]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"rank {r}\n(GPU {r})" for r in xs]
    b1 = ax.bar(labels, compute, color="#2b7bba", label="GPU compute (conv/gemm/bn/elementwise)")
    b2 = ax.bar(labels, comm, bottom=compute, color="#e8743b",
                label="RCCL communication (ncclDevKernel all-reduce)")

    for i, r in enumerate(xs):
        tot = compute[i] + comm[i]
        frac = 100.0 * comm[i] / tot if tot else 0.0
        ax.text(i, tot, f"comm {comm[i]:.2f}s\n({frac:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("GPU kernel exclusive time (s)")
    ax.set_title(args.title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)
    for r in xs:
        c, m = ranks[r]["compute"] / 1000.0, ranks[r]["comm"] / 1000.0
        print(f"  rank {r}: compute={c:.3f}s comm={m:.3f}s comm%={100*m/(c+m):.1f}")


if __name__ == "__main__":
    main()
