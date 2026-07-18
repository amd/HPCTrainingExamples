#!/usr/bin/env python3
"""Render a Score-P summary figure for the CG-GPU example.

Parses the text output committed alongside this script:
  * cg_scorep_score.txt  -- `scorep-score` region-type breakdown (ALL/HIP/MPI/...)
  * cg_cube_stat.txt      -- `cube_stat -p -m time` flat per-routine time profile

and writes scorep_cg_breakdown.png:
  (left)  time by region type (HIP vs MPI vs measurement overhead)
  (right) the top routines by time, coloured by class (HIP / MPI)

Reproduce:
  python -m venv --system-site-packages ~/scorep-venvs/figs
  source ~/scorep-venvs/figs/bin/activate && pip install matplotlib
  python make_scorep_fig.py
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_score(path):
    """Return {type: time_s} for the ALL/HIP/MPI/... rows of scorep-score."""
    out = {}
    for line in open(path):
        m = re.match(r"\s*(?:\*?)\s*(ALL|HIP|MPI|OMP|USR|COM|SCOREP)\s+"
                     r"[\d,]+\s+[\d,]+\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_cube_stat(path):
    """Return [(routine, time_s)] from the flat cube_stat profile."""
    rows = []
    for line in open(path):
        m = re.match(r"\s+([A-Za-z_][\w:().]*)\s+([\d.]+)\s*$", line)
        if m:
            name, t = m.group(1), float(m.group(2))
            if name.startswith(("INCL", "EXCL")):
                continue
            rows.append((name, t))
    return rows


def classify(name):
    if name.startswith("MPI_"):
        return "MPI", "#1f77b4"
    if name.startswith(("hip", "__hip", "KERNELS", "THREADS")):
        return "HIP/GPU", "#d62728"
    return "other", "#7f7f7f"


def main():
    score = parse_score(os.path.join(HERE, "cg_scorep_score.txt"))
    rows = parse_cube_stat(os.path.join(HERE, "cg_cube_stat.txt"))
    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[:12][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: region-type time breakdown from scorep-score.
    order = [k for k in ("HIP", "MPI", "SCOREP") if k in score]
    labels = {"HIP": "HIP / GPU", "MPI": "MPI", "SCOREP": "measurement\noverhead"}
    colors = {"HIP": "#d62728", "MPI": "#1f77b4", "SCOREP": "#7f7f7f"}
    vals = [score[k] for k in order]
    bars = ax1.bar([labels[k] for k in order], vals,
                   color=[colors[k] for k in order])
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}s",
                 ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("aggregate time across ranks (s)")
    ax1.set_title("Score-P region-type breakdown\n(scorep-score)")
    ax1.margins(y=0.15)

    # Right: top routines by time.
    names = [r[0] for r in top]
    times = [r[1] for r in top]
    bcolors = [classify(n)[1] for n in names]
    ax2.barh(range(len(top)), times, color=bcolors)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("inclusive time (s)")
    ax2.set_title("Top routines by time\n(cube_stat -p -m time)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in ("#1f77b4", "#d62728")]
    ax2.legend(handles, ["MPI", "HIP / GPU"], loc="lower right")

    fig.suptitle("CG-GPU under Score-P  -  method=isend, 4 ranks, "
                 "Dubcova2.pm  (ROCm 6.4.3, MI300A)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "scorep_cg_breakdown.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
