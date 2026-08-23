#!/usr/bin/env python3
"""Render the TraceLens FSDP2 results (real data) into a single figure.

Reads the collective-analysis and gpu_timeline numbers produced by
  TraceLens_generate_multi_rank_collective_report_pytorch  (nccl_csvs/)
  TraceLens_generate_perf_report_pytorch                   (gpu_timeline sheet)
and draws: (1) the GPU-time split, (2) per-collective achieved bus bandwidth,
(3) comm latency vs. inter-rank start skew (the scaling story).

Usage: python make_tracelens_fig.py [out.png]
Numbers below are the measured MI300A / 4-GPU / fp32 run; regenerate the CSVs
with submit_timeline_traces.sbatch and update if you re-measure.
"""
import sys, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "figs/fsdp2_tracelens_collective.png"

# --- gpu_timeline (rank 0, ms) ---
tl = [("computation", 755.09), ("exposed comm", 135.59), ("idle", 53.35)]
tl_total_comm = 419.02  # total comm incl. overlapped

# --- per-collective (implicit-sync summary) ---
# (label, msg MB, count, comm_latency_us, start_skew_us, bus_bw_GBs)
coll = [
    ("all-gather\n48 MB", 48.0, 192, 396.5, 1162.5, 102.1),
    ("reduce-scatter\n48 MB", 48.0, 96, 245.8, 925.2, 144.0),
    ("all-gather\n252 MB", 252.0, 6, 827.4, 9748.6, 223.1),
    ("reduce-scatter\n252 MB", 252.0, 6, 866.9, 1253.4, 212.9),
]

# If a fresh collective CSV is present, use its measured numbers instead.
_csv = "torch_prof/nccl_csvs/nccl_summary_implicit_sync.csv"
if os.path.exists(_csv):
    rows = []
    with open(_csv) as f:
        for r in csv.DictReader(f):
            name = r["Collective name"]
            if name not in ("_allgather_base", "_reduce_scatter_base"):
                continue
            mb = float(r["Full msg size (MB)"])
            short = "all-gather" if "allgather" in name else "reduce-scatter"
            rows.append((f"{short}\n{mb:.0f} MB", mb, int(float(r["count"])),
                         float(r["comm_latency_mean"]),
                         float(r["skew in start time_mean"]),
                         float(r["bus bw (GB/s)_mean"])))
    if rows:
        coll = sorted(rows, key=lambda x: (x[1], x[0]))
labels = [c[0] for c in coll]
lat = [c[3] for c in coll]
skew = [c[4] for c in coll]
bus = [c[5] for c in coll]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# Panel 1: GPU time split
left = 0
for name, ms in tl:
    ax[0].barh(0, ms, left=left, label=f"{name} ({ms:.0f} ms)")
    left += ms
ax[0].set_yticks([])
ax[0].set_xlabel("GPU time (ms)")
ax[0].set_title("gpu_timeline: 80% compute, 14% exposed comm\n"
                f"(total comm {tl_total_comm:.0f} ms — ~70% hidden behind compute)",
                fontsize=9)
ax[0].legend(loc="lower center", fontsize=8, ncol=1)

# Panel 2: bus bandwidth
x = range(len(labels))
ax[1].bar(x, bus, color=["#4C72B0", "#55A868", "#4C72B0", "#55A868"])
ax[1].set_xticks(list(x)); ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_ylabel("bus bandwidth (GB/s)")
ax[1].set_title("Achieved RCCL bus bandwidth per collective\n"
                "(compare to rccl-tests ceiling)", fontsize=9)
for i, v in enumerate(bus):
    ax[1].text(i, v + 3, f"{v:.0f}", ha="center", fontsize=8)

# Panel 3: latency vs skew (log scale — skew dominates)
w = 0.38
ax[2].bar([i - w/2 for i in x], lat, w, label="comm latency (µs)", color="#4C72B0")
ax[2].bar([i + w/2 for i in x], skew, w, label="start skew (µs)", color="#C44E52")
ax[2].set_yscale("log")
ax[2].set_xticks(list(x)); ax[2].set_xticklabels(labels, fontsize=7)
ax[2].set_ylabel("µs (log)")
ax[2].set_title("Inter-rank skew >> comm latency\n= skew/imbalance-bound, not link-bound",
                fontsize=9)
ax[2].legend(fontsize=8)

fig.suptitle("TraceLens — FSDP2 transformer (MI300A, 4 GPUs, fp32): RCCL collective analysis",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(out, dpi=130)
print("wrote", out)
