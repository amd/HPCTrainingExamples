#!/usr/bin/env python3
"""Render a KCachegrind-style call/cost graph from a Valgrind *callgrind* output
file as a PNG, fully headless (matplotlib Agg) so it works over plain SSH with no
X server / KCachegrind install.

Cachegrind output has no caller->callee edges, so for the *call graph* we use a
companion callgrind run (same cache model, plus the call graph). Node label shows
the (demangled, shortened) function name with self% and inclusive% of total Ir
(instructions). Edge labels show call counts; edge width scales with inclusive
cost carried along that call.

Usage:  python3 render_callgraph.py <callgrind.out> <out.png> [top_n]
"""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def parse_callgrind(path):
    """Return (self_ir, edges, names) where self_ir[fn]=self Ir, edges[(a,b)]=
    (calls, inclusive_ir), names is the set of functions."""
    fn_dict = {}
    events = None
    ir_idx = 0
    num_pos = 1  # positions: default is "line" -> 1 leading position column
    prog_total = 0
    cur_fn = None
    self_ir = {}
    edges = {}
    pending_call = None  # (callee, ncalls) awaiting its inclusive-cost line

    def resolve_fn(token):
        # token like "(12) name"  or "(12)"  or "name"
        m = re.match(r"\((\d+)\)\s*(.*)", token)
        if m:
            key, name = m.group(1), m.group(2).strip()
            if name:
                fn_dict[key] = name
            return fn_dict.get(key, key)
        return token.strip()

    def cost_ir(tokens):
        # layout is: <num_pos position cols> <event cols...>; callgrind drops
        # trailing zero event cols, so index Ir from the FRONT (after positions).
        try:
            return int(tokens[num_pos + ir_idx])
        except (ValueError, IndexError):
            return 0

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("events:"):
                events = line.split()[1:]
                if "Ir" in events:
                    ir_idx = events.index("Ir")
                continue
            if line.startswith("positions:"):
                num_pos = max(1, len(line.split()[1:]))
                continue
            if line.startswith("summary:") or line.startswith("totals:"):
                vals = line.split()[1:]
                if vals:
                    try:
                        prog_total = int(vals[ir_idx])
                    except (ValueError, IndexError):
                        pass
                continue
            if line.startswith("fl=") or line.startswith("fi=") or line.startswith("fe="):
                continue
            if line.startswith("fn="):
                cur_fn = resolve_fn(line[3:])
                self_ir.setdefault(cur_fn, 0)
                pending_call = None
                continue
            if line.startswith("cfn="):
                pending_call = [resolve_fn(line[4:]), 0]
                continue
            if line.startswith("cfi=") or line.startswith("cob=") or line.startswith("cfl="):
                continue
            if line.startswith("calls="):
                m = re.match(r"calls=(\d+)", line)
                if m and pending_call is not None:
                    pending_call[1] = int(m.group(1))
                continue
            if line.startswith("#") or "=" in line.split(" ", 1)[0]:
                # header / directive we don't need (positions:, summary:, etc.)
                continue
            # otherwise a cost line: positions... cost0 cost1 ...
            if events is None:
                continue
            toks = line.split()
            ir = cost_ir(toks)
            if pending_call is not None and cur_fn is not None:
                callee, ncalls = pending_call
                a, b = cur_fn, callee
                c, inc = edges.get((a, b), (0, 0))
                edges[(a, b)] = (c + ncalls, inc + ir)
                pending_call = None
            elif cur_fn is not None:
                self_ir[cur_fn] = self_ir.get(cur_fn, 0) + ir

    names = set(self_ir) | {b for (_, b) in edges} | {a for (a, _) in edges}
    for nm in names:
        self_ir.setdefault(nm, 0)
    if not prog_total:
        prog_total = sum(self_ir.values())
    return self_ir, edges, names, prog_total


def shorten(name):
    """Trim C++ signatures/namespaces to something legible."""
    name = name.split("(")[0]
    name = name.replace("std::", "").replace("__gnu_cxx::", "")
    if "::" in name:
        name = "::".join(name.split("::")[-2:])
    return name[:34]


SOLVER_KEYS = ("spmv(", "inner_product(", "axpy(", "scale(", "readParMatrix(",
               "operator=")


def is_interesting(name):
    """Keep the CG solver functions + the libc bulk-copy/set they drive; drop the
    MPI/PMIx/hwloc singleton-init and libc-startup noise callgrind also records."""
    if name == "main":
        return True
    if "memcpy" in name or "memset" in name:
        return True
    return any(k in name for k in SOLVER_KEYS)


def inclusive_costs(self_ir, edges, names):
    inc = dict(self_ir)
    for (a, b), (c, e) in edges.items():
        inc[a] = inc.get(a, 0) + e
    return inc


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 14

    self_ir, edges, names, prog_total = parse_callgrind(src)
    total = max(prog_total, 1)

    # keep the CG solver functions (drop MPI/PMIx/hwloc init + libc startup noise)
    keep = set(n for n in names if is_interesting(n))
    keep = set(sorted(keep, key=lambda n: self_ir.get(n, 0), reverse=True)[:top_n])
    kept_edges = {(a, b): v for (a, b), v in edges.items()
                  if a in keep and b in keep and a != b}
    # drop disconnected near-zero leaves (duplicate libc memcpy/memset stubs)
    deg = {}
    for (a, b) in kept_edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    keep = {n for n in keep
            if deg.get(n, 0) > 0 or 100.0 * self_ir.get(n, 0) / max(prog_total, 1) >= 1.0}
    kept_edges = {(a, b): v for (a, b), v in kept_edges.items()
                  if a in keep and b in keep}

    # inclusive over the kept subgraph: callgrind's per-call cost is already the
    # full callee-subtree inclusive Ir, so inclusive[caller] = self + sum(edge_inc)
    inc = {n: self_ir.get(n, 0) for n in keep}
    for (a, b), (c, e) in kept_edges.items():
        inc[a] += e

    # layer assignment: shortest path (BFS depth) from roots -> compact tree
    succ = {}
    preds = {n: [] for n in keep}
    indeg = {n: 0 for n in keep}
    for (a, b) in kept_edges:
        succ.setdefault(a, []).append(b)
        preds[b].append(a)
        indeg[b] = indeg.get(b, 0) + 1
    roots = [n for n in keep if indeg.get(n, 0) == 0] or [max(keep, key=lambda n: inc.get(n, 0))]
    depth = {}
    from collections import deque
    dq = deque((r, 0) for r in roots)
    for r in roots:
        depth[r] = 0
    while dq:
        n, d = dq.popleft()
        for m in succ.get(n, []):
            if m not in depth or depth[m] > d + 1:
                depth[m] = d + 1
                dq.append((m, d + 1))
    # any node not reached (no incoming from roots) -> place after its deepest pred
    for n in keep:
        if n not in depth:
            depth[n] = 1 + max((depth.get(p, 0) for p in preds[n]), default=0)

    layers = {}
    for n in keep:
        layers.setdefault(depth[n], []).append(n)
    for lv in layers:
        layers[lv].sort(key=lambda n: -inc.get(n, 0))

    # positions
    max_per = max(len(v) for v in layers.values())
    pos = {}
    ncols = max(layers) + 1
    for lv, nodes in layers.items():
        for i, n in enumerate(nodes):
            x = lv * 3.0
            y = -(i - (len(nodes) - 1) / 2.0) * 1.7
            pos[n] = (x, y)

    fig_w = max(9, ncols * 3.1)
    fig_h = max(5.5, max_per * 1.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    cmap = plt.get_cmap("YlOrRd")

    # edges first
    max_e = max((e for (_, e) in kept_edges.values()), default=1) or 1
    for (a, b), (c, e) in kept_edges.items():
        xa, ya = pos[a]
        xb, yb = pos[b]
        lw = 0.8 + 4.5 * (e / max_e)
        arr = FancyArrowPatch((xa + 1.15, ya), (xb - 1.15, yb),
                              arrowstyle="-|>", mutation_scale=16,
                              lw=lw, color="#555555", alpha=0.75,
                              connectionstyle="arc3,rad=0.06", zorder=1)
        ax.add_patch(arr)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my + 0.12, f"{c:,}x", fontsize=7.5, color="#1a1a1a",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                zorder=3)

    # nodes
    for n in keep:
        x, y = pos[n]
        s_pct = 100.0 * self_ir.get(n, 0) / total
        i_pct = 100.0 * inc.get(n, 0) / total
        color = cmap(min(0.92, 0.12 + 0.85 * (self_ir.get(n, 0) / (max(self_ir.values()) or 1))))
        box = FancyBboxPatch((x - 1.1, y - 0.55), 2.2, 1.1,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc=color, ec="#333333", lw=1.2, zorder=2)
        ax.add_patch(box)
        ax.text(x, y + 0.22, shorten(n), fontsize=8.5, fontweight="bold",
                ha="center", va="center", zorder=3)
        ax.text(x, y - 0.20, f"self {s_pct:4.1f}%  |  incl {i_pct:4.1f}%",
                fontsize=7.5, ha="center", va="center", color="#222222", zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.8, max(xs) + 1.8)
    ax.set_ylim(min(ys) - 1.4, max(ys) + 1.6)
    ax.set_title("CG-CPU call graph (Valgrind callgrind, Ir = instructions)\n"
                 "node shade = self cost; edge width = inclusive cost; label = call count",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({len(keep)} nodes, {len(kept_edges)} edges, total Ir={total:,})")


if __name__ == "__main__":
    main()
