"""torchrun shim: bind each rank to its GPU's local NUMA node, then exec the script.

On MI300A (SPX) the GPU<->NUMA mapping is 1:1 (GPU i -> node i), each node
holding 24 cores + 128 GB local HBM. Without binding, ranks float across all 4
sockets and host allocations can land on remote HBM (distance 32 vs 10 local).
For the GPU-resident synthetic benchmarks here the effect is within noise (see
common/PERFORMANCE_NOTES.md), but binding matters for CPU-heavy input pipelines,
host-staged RCCL fallback (NCCL_P2P_DISABLE=1), multi-node NIC locality, and
CPU-offload optimizers -- hence this optional, drop-in launcher.

Usage (drop-in in front of the real script):
    torchrun --standalone --nproc_per_node=4 affinity_launcher.py REAL.py [args...]

Or via the sweep drivers:  AFFINITY=1 ./rccl_scaling_sweep.sh

It re-execs the *same* PID under numactl so torchrun still tracks the process and
all torchrun env vars (RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*) are inherited. If
numactl is missing it falls back to a pure-Python CPU pin (memory locality then
relies on first-touch) so the run still proceeds.

Override the GPU->node map with AFFINITY_NODES="0,1,2,3" (index by LOCAL_RANK).
"""
import os
import shutil
import sys


def _cpu_pin_fallback(node):
    """Best-effort CPU affinity when numactl is unavailable (no membind)."""
    try:
        path = f"/sys/devices/system/node/node{node}/cpulist"
        with open(path) as f:
            spec = f.read().strip()
        cpus = set()
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                cpus.update(range(int(a), int(b) + 1))
            elif part:
                cpus.add(int(part))
        if cpus:
            os.sched_setaffinity(0, cpus)
    except Exception:
        pass


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: affinity_launcher.py REAL_SCRIPT.py [args...]")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    nodes = os.environ.get("AFFINITY_NODES")
    node = (int(nodes.split(",")[local_rank]) if nodes else local_rank)

    if shutil.which("numactl"):
        argv = ["numactl", f"--cpunodebind={node}", f"--membind={node}",
                sys.executable] + sys.argv[1:]
        os.execvp("numactl", argv)
    else:
        _cpu_pin_fallback(node)
        os.execvp(sys.executable, [sys.executable] + sys.argv[1:])


if __name__ == "__main__":
    main()
