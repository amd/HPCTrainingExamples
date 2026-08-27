#!/usr/bin/env python3
"""Render the CG-CPU classic roofline with AMD uProf's own AMDuProfModelling.py.

This is a thin driver: it imports uProf's shipped modelling script unmodified
(so every roof and the plotted point come from uProf's model) and only forces
matplotlib to save with a tight bounding box + a PNG copy, so the right-hand
roof labels are not clipped.  All numbers come from the input CSV
(measured FP throughput from AMDuProfPcm + DRAM byte traffic from cachegrind).

Usage:
  python3 render_cpu_roofline.py <roofline.csv> <out_dir> <png_path> "<app label>"
"""
import importlib.util
import os
import sys

UPROF = "/nfsapps/ubuntu-24.04-nightlies/opt/AMDuProf_5.3-518/bin/AMDuProfModelling.py"

def main():
    csv, out_dir, png, label = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)

    spec = importlib.util.spec_from_file_location("uprofmod", UPROF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module name != __main__, so main() does not auto-run

    from matplotlib import pyplot as plt
    orig_savefig = plt.savefig

    def savefig_tight(fname, *a, **k):
        k["bbox_inches"] = "tight"
        k.setdefault("dpi", 150)
        orig_savefig(fname, *a, **k)          # uProf's PDF (tight)
        orig_savefig(png, bbox_inches="tight", dpi=150)  # PNG for the docs
        print("Wrote", png)

    plt.savefig = savefig_tight

    sys.argv = ["AMDuProfModelling.py", "-i", csv, "-o", out_dir,
                "-S", "-p", "roofline", "-c", "1", "--dp-roofs", "-a", label]
    mod.main()

if __name__ == "__main__":
    main()
