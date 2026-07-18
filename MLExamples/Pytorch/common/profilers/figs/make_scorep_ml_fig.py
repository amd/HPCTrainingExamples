#!/usr/bin/env python3
"""Render a Score-P summary figure for the three PyTorch examples.

Parses the committed `*_cube_stat.txt` (from `cube_stat -p -m time` on rank 0 of
each instrumented run) and writes scorep_ml_breakdown.png: for each example, the
wall time Score-P attributed to the hand-placed **training-step user regions** vs.
the remaining Python time (import torch / model build / dist init / first-step
MIOpen autotune), which lands outside the regions.

Reproduce:
  source ~/scorep-venvs/figs/bin/activate   # a venv with matplotlib
  python make_scorep_ml_fig.py
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [("imagenet", "imagenet\n(resnet50, DDP)"),
            ("minGPT", "minGPT-ddp\n(GPT, DDP)"),
            ("fsdp2", "FSDP2\n(sharded)")]


def parse(name):
    """Return (training_region_s, other_python_s) from a cube_stat file."""
    training, other = 0.0, 0.0
    path = os.path.join(HERE, f"{name}_cube_stat.txt")
    for line in open(path):
        m = re.match(r"\s*EXCL\(python[\d.]*\)\s+([\d.]+)", line)
        if m:
            other = float(m.group(1))
        m = re.match(r"\s*user:\S+\s+([\d.]+)", line)
        if m:
            training += float(m.group(1))
    return training, other


def main():
    labels = [lbl for _, lbl in EXAMPLES]
    train = [parse(k)[0] for k, _ in EXAMPLES]
    other = [parse(k)[1] for k, _ in EXAMPLES]

    x = range(len(EXAMPLES))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x, train, color="#2ca02c", label="training-step user regions")
    b2 = ax.bar(x, other, bottom=train, color="#c7c7c7",
                label="other Python (import/build/init/autotune)")
    for i, (t, o) in enumerate(zip(train, other)):
        ax.text(i, t / 2, f"{t:.1f}s", ha="center", va="center", fontsize=10)
        ax.text(i, t + o / 2, f"{o:.1f}s", ha="center", va="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("rank-0 wall time (s)")
    ax.set_title("Score-P user-region time across the PyTorch examples\n"
                 "(2 GPUs, short run; --nopython, ROCm 7.2.3 / MI300A)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(HERE, "scorep_ml_breakdown.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
