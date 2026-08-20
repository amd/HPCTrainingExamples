#!/usr/bin/env python3
"""Render a likwid roofline for the CG-CPU solver on one MI300A APU core.

With likwid >= 5.5.2 (MI300A support, `module load likwid`) the classic
"likwid-bench for the roofs, likwid-perfctr for the point" workflow now works
for compute and the cache levels; only the DRAM (UMC uncore) counters are still
unreadable on this node.  We therefore build the roofline from:
  * ROOFS  -> likwid-bench (works): single-core compute peaks + L1/L2/L3/DRAM
             read bandwidth.  See measure_likwid_bench.sh.
  * POINT  -> likwid-perfctr native counters where available:
               FLOPS_DP -> retired SSE/AVX FLOPs (compute rate),
               L2 / L3  -> data volume moved at each cache level (AI vs L2, L3).
             The MEM/MEM_DP (UMC) group still reads zero here (no amd_umc/amd_df
             perf PMU and no /dev/cpu/*/msr on the node), so the L1 and DRAM byte
             traffic still come from cachegrind.

All numbers are single-core, double precision, Dubcova2, seed 12345.
Edit the dicts below to match a fresh measurement run.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- likwid-bench ceilings (single core), MByte/s -> GB/s, MFlop/s -> GF/s ---
COMPUTE = {            # horizontal roofs (GFLOP/s)
    "AVX-512 FMA peak": 59.0,
    "scalar peak":       9.5,
}
MEMBW = {              # diagonal roofs (GB/s), likwid-bench load_avx512 / triad
    "L1":   213.0,
    "L2":   118.0,
    "L3":    92.0,
    "DRAM":  21.6,
}

# --- measured cg_cpu point ---------------------------------------------------
# Compute rate: likwid-perfctr -g FLOPS_DP (RETIRED_SSE_AVX_FLOPS_ALL) over the
# CG solve region (seed 12345, Dubcova2, 172 iters).
FLOPS_G   = 0.5778     # retired SSE/AVX FLOPs (GFLOP), likwid FLOPS_DP
SOLVE_S   = 0.163      # CG solve time (s)
TP        = FLOPS_G / SOLVE_S           # achieved GFLOP/s (~3.5)
# bytes moved at each level; source noted per level.
BYTES = {              # level -> (bytes, source)
    "L1":   (1.989e9 * 8, "cachegrind"),   # D refs x 8 B  (approx; mixed 4/8 B)
    "L2":   (5.9626e9,    "likwid L2"),    # likwid-perfctr L2 data volume
    "L3":   (4.8866e9,    "likwid L3"),    # likwid-perfctr L3 data volume
    "DRAM": (657223.0 * 64, "cachegrind"), # LLd misses x 64 B (UMC unreadable)
}

def main():
    png = sys.argv[1] if len(sys.argv) > 1 else "cg_likwid_roofline_cpu_n1.png"
    peak = max(COMPUTE.values())

    fig, ax = plt.subplots(figsize=(9, 6))
    ai = np.logspace(-3, 2, 500)

    # diagonal memory roofs, clipped at the top compute peak
    mem_colors = {"L1": "#1f77b4", "L2": "#2ca02c", "L3": "#ff7f0e", "DRAM": "#d62728"}
    for name, bw in MEMBW.items():
        y = np.minimum(bw * ai, peak)
        ax.plot(ai, y, color=mem_colors[name], lw=1.5)
        knee = peak / bw
        ax.text(knee * 0.62, peak * 1.03, f"{name} {bw:.0f} GB/s",
                color=mem_colors[name], fontsize=8, rotation=38,
                ha="right", va="bottom")

    # horizontal compute roofs
    for name, gf in COMPUTE.items():
        ls = "-" if gf == peak else "--"
        ax.hlines(gf, MEMBW["L1"] and (gf / MEMBW["L1"]), 1e2,
                  color="black", lw=1.2, ls=ls)
        ax.text(1e2, gf * 1.03, f"{name} {gf:.1f} GF/s",
                color="black", fontsize=9, ha="right", va="bottom")

    # cg_cpu point at each memory level's arithmetic intensity.
    # circle 'o' = native likwid-perfctr counter; diamond 'D' = cachegrind.
    pt_colors = {"L1": "#1f77b4", "L2": "#2ca02c", "L3": "#ff7f0e", "DRAM": "#d62728"}
    for lvl, (b, src) in BYTES.items():
        x = FLOPS_G * 1e9 / b
        mk = "o" if src.startswith("likwid") else "D"
        ax.scatter([x], [TP], s=75, zorder=5, marker=mk, color=pt_colors[lvl],
                   edgecolor="black", linewidth=0.6,
                   label=f"cg_cpu vs {lvl} ({src}): AI={x:.3g} FLOP/B, {TP:.1f} GF/s")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e2); ax.set_ylim(0.5, peak * 2.2)
    ax.set_xlabel("Arithmetic intensity [FLOP/Byte]")
    ax.set_ylabel("Throughput [GFLOP/s]")
    ax.set_title("likwid-bench roofline — MI300A APU CPU core (1 core, DP)\n"
                 "cg_cpu on Dubcova2 (172 iters)")
    ax.grid(True, which="both", lw=0.25, color="gray")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print("Wrote", png)

if __name__ == "__main__":
    main()
