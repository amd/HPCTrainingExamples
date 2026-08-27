#!/usr/bin/env python3
"""Render the per-rank GPU-time figure (compute vs RCCL all-reduce) from TAU.

Fully headless (matplotlib 'Agg') -- no Java, ParaProf, or X server needed.
Input is the output of `pprof -a` (see profiling/tau/tau_pprof.txt), which lists,
per NODE;CONTEXT;THREAD, the exclusive time of every GPU kernel. We aggregate the
GPU-device kernels ("[ROCm Kernel] ...") per rank (NODE) and split them into:
  * communication : RCCL/NCCL collectives (ncclDevKernel*)
  * compute       : every other GPU kernel (conv/gemm/batchnorm/elementwise/...)

The figure is a single panel: per-rank stacked bars of GPU-kernel exclusive time,
split into compute (blue) and the RCCL all-reduce (orange, ncclDevKernel). A shaded
band spans the fastest-to-slowest rank's compute so the compute-time spread is visible,
and each bar is annotated with the all-reduce fraction of that rank's GPU time.

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


COMPUTE_COLOR = "#2b7bba"
COMM_COLOR = "#e8743b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pprof")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title",
                    default="TAU (ParaProf profile): 4-rank ImageNet on MI300A -- "
                            "per-rank GPU time (compute vs RCCL all-reduce)")
    args = ap.parse_args()

    ranks = parse(args.pprof)
    if not ranks:
        sys.exit("no per-rank GPU kernel data found in " + args.pprof)

    xs = sorted(ranks)
    compute = [ranks[r]["compute"] / 1000.0 for r in xs]  # -> seconds
    comm = [ranks[r]["comm"] / 1000.0 for r in xs]
    labels = [f"rank {r}\n(GPU {r})" for r in xs]

    cmin, cmax = min(compute), max(compute)
    cspread = cmax - cmin
    cspread_pct = 100.0 * cspread / cmax if cmax else 0.0

    fig, axA = plt.subplots(1, 1, figsize=(9, 5.5))

    # -- Stacked compute + all-reduce per rank -----------------------------------
    axA.bar(labels, compute, color=COMPUTE_COLOR,
            label="GPU compute (conv/gemm/bn/elementwise)")
    axA.bar(labels, comm, bottom=compute, color=COMM_COLOR,
            label="RCCL all-reduce time\n(reduction + wait, ncclDevKernel)")

    # Shade the compute-imbalance band [min compute, max compute] across all ranks.
    # The band is covered by the (taller) bars, so label it in the empty top strip
    # rather than with an arrow that would cross the bars.
    axA.axhspan(cmin, cmax, color=COMPUTE_COLOR, alpha=0.12, zorder=0)
    axA.axhline(cmin, color=COMPUTE_COLOR, ls="--", lw=1.2, alpha=0.9,
                label=f"fastest / slowest compute ({cmin:.2f}s / {cmax:.2f}s)")
    axA.axhline(cmax, color=COMPUTE_COLOR, ls="--", lw=1.2, alpha=0.9)
    tot_max = max(compute[i] + comm[i] for i in range(len(xs)))
    axA.text(len(xs) - 1, tot_max * 1.16,
             f"compute-kernel time spread:\n\u0394 = {cspread:.2f}s ({cspread_pct:.0f}% of max)",
             ha="center", va="center", fontsize=9.5, color=COMPUTE_COLOR,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COMPUTE_COLOR, alpha=0.9))

    for i in range(len(xs)):
        tot = compute[i] + comm[i]
        frac = 100.0 * comm[i] / tot if tot else 0.0
        axA.text(i, compute[i] / 2.0, f"{compute[i]:.2f}s",
                 ha="center", va="center", fontsize=8, color="white")
        axA.text(i, tot, f"comm {comm[i]:.2f}s\n({frac:.1f}%)",
                 ha="center", va="bottom", fontsize=8)

    axA.set_ylabel("GPU kernel exclusive time (s)")
    axA.set_title("Per-rank GPU time: compute vs RCCL all-reduce")
    axA.legend(loc="upper left", fontsize=8)
    axA.grid(axis="y", ls=":", alpha=0.5)
    axA.margins(y=0.30)

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)

    print("wrote", args.out)
    for r, c, m in zip(xs, compute, comm):
        print(f"  rank {r}: compute={c:.3f}s comm={m:.3f}s comm%={100*m/(c+m):.1f}")
    print(f"  compute-kernel spread: min={cmin:.3f}s max={cmax:.3f}s "
          f"spread={cspread:.3f}s ({cspread_pct:.1f}% of max)")


if __name__ == "__main__":
    main()
