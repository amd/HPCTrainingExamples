#!/usr/bin/env python3
"""
CuPy Unified Memory Programming (UMP) micro-benchmarks

Two isolated subprocess runs:
  1) Baseline: CUPY_ENABLE_UMP=0
  2) UMP:      CUPY_ENABLE_UMP=1 (+ malloc_system allocator; optional numpy_allocator)

Measures per-size averages for:
  - HtoD: cp.asarray(np_arr)
  - DtoH: cp_arr.get()
  - Add:  x += y   (device compute)
  - Pipeline: np -> cp -> (x += y) -> np

Usage:
  python cupy_ump_bench.py
  python cupy_ump_bench.py --sizes 16MiB 256MiB 1GiB --repeats 100 --warmups 5 --dtype float32 --csv out.csv
"""

from __future__ import annotations
import argparse, os, sys, time, json, subprocess
from typing import List, Dict


def clear_gpu():
    import cupy as cp, gc
    cp.cuda.Stream.null.synchronize()  
    gc.collect()                       
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def parse_size_to_bytes(token: str) -> int:
    s = token.strip().lower()
    units = {"b":1,"kib":1024,"kb":1000,"mib":1024**2,"mb":1000**2,"gib":1024**3,"gb":1000**3}
    for u in ("gib","gb","mib","mb","kib","kb","b"):
        if s.endswith(u):
            return int(float(s[:-len(u)]) * units[u])
    return int(s)

def human_bytes(n: int) -> str:
    for name, div in (("GiB",1024**3),("MiB",1024**2),("KiB",1024)):
        if n >= div: return f"{n/div:.2f} {name}"
    return f"{n} B"

def dtype_from_str(name: str):
    import cupy as cp
    try: return getattr(cp, name)
    except AttributeError: raise SystemExit(f"Unsupported dtype: {name}")

def sync():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()

def now() -> float:
    return time.perf_counter()

def run_worker(mode: str, sizes_b: List[int], repeats: int, warmups: int, dtype_name: str) -> Dict:
    import numpy as np
    import cupy as cp

   
    if mode == "ump":
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.memory.malloc_system).malloc)
        
        import cupy._core.numpy_allocator as ac
        import numpy_allocator
        import ctypes        
        lib = ctypes.CDLL(ac.__file__)
        class my_allocator(metaclass=numpy_allocator.type):
            _calloc_ = ctypes.addressof(lib._calloc)
            _malloc_ = ctypes.addressof(lib._malloc)
            _realloc_ = ctypes.addressof(lib._realloc)
            _free_ = ctypes.addressof(lib._free)
        my_allocator.__enter__()
        
       

    dtype = dtype_from_str(dtype_name)
    itemsize = int(cp.dtype(dtype).itemsize)

    
    _ = (cp.arange(16, dtype=cp.float32) + 1).sum()
    sync()

    out = []
    for sz in sizes_b:
        clear_gpu()
        n = sz // itemsize
        if n == 0: continue
        rec = {"size_bytes": int(sz), "n_elems": int(n)}

        
        h = np.random.random(n).astype(dtype)
        for _ in range(warmups):
            _ = cp.asarray(h); sync()
        t = 0.0
        for _ in range(repeats):
            t0 = now()
            x = cp.asarray(h)
            sync()
            t += (now() - t0)
            del x
        rec["HtoD_asarray_ms"] = (t / repeats) * 1000.0
        del h
        
        x = cp.random.random(n, dtype=dtype); sync()
        for _ in range(warmups):
            _ = x.get(); sync()
        t = 0.0
        for _ in range(repeats):
            t0 = now()
            _ = x.get()
            sync()
            t += (now() - t0)
        rec["DtoH_get_ms"] = (t / repeats) * 1000.0
        del x
        
        x = cp.random.random(n, dtype=dtype)
        y = cp.random.random(n, dtype=dtype)
        sync()
        for _ in range(warmups):
            x += y; sync()
        t = 0.0
        for _ in range(repeats):
            t0 = now()
            x += y
            sync()
            t += (now() - t0)
        rec["add_inplace_ms"] = (t / repeats) * 1000.0
        del x,y
        
        a = np.random.random(n).astype(dtype)
        b = np.random.random(n).astype(dtype)
        for _ in range(warmups):
            xa = cp.asarray(a); yb = cp.asarray(b); xa += yb; _ = xa.get(); sync()
        t = 0.0
        for _ in range(repeats):
            t0 = now()
            xa = cp.asarray(a)
            yb = cp.asarray(b)
            xa += yb
            _ = xa.get()
            sync()
            t += (now() - t0)
        rec["pipeline_np_cp_compute_np_ms"] = (t / repeats) * 1000.0
        del a, b, xa, yb

        def gibps(bytes_amt, ms):
            return float("nan") if ms <= 0 else (bytes_amt / (1024**3)) / (ms / 1000.0)
        rec["HtoD_GiBps"] = gibps(sz, rec["HtoD_asarray_ms"])
        rec["DtoH_GiBps"] = gibps(sz, rec["DtoH_get_ms"])

        out.append(rec)
        clear_gpu()
    
    try:
        if numpy_alloc_guard is not None:
            numpy_alloc_guard.__exit__(None, None, None)  # restore NumPy’s allocator
        cp.cuda.set_allocator(None)  # back to default pool allocator
        clear_gpu()
    except Exception:
        pass

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    meta = {
        "mode": mode, 
        "device": {
            "name": props["name"].decode() if isinstance(props["name"], (bytes, bytearray)) else props["name"],
            "multiProcessorCount": int(props["multiProcessorCount"]),
            "totalGlobalMem": int(props["totalGlobalMem"]),
        },
        "dtype": dtype_name,
        "repeats": repeats,
        "warmups": warmups,
    }
    return {"meta": meta, "results": out}

def main():
    p = argparse.ArgumentParser(description="CuPy UMP benchmark (baseline vs UMP).")
    p.add_argument("--sizes", nargs="*", default=["16MiB", "256MiB", "1GiB"],
                   help="Sizes like: 8MiB 64MiB 1GiB (default: 16MiB 256MiB 1GiB)")
    p.add_argument("--repeats", type=int, default=100, help="Repeats per measurement (default: 100)")
    p.add_argument("--warmups", type=int, default=3, help="Warmup iterations per measurement (default: 3)")
    p.add_argument("--dtype", default="float32", help="CuPy dtype name (default: float32)")
    p.add_argument("--csv", default=None, help="Optional CSV path")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_mode", choices=["baseline","ump"], help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._worker:
        sizes_b = [parse_size_to_bytes(s) for s in args.sizes]
        payload = run_worker(args._mode, sizes_b, args.repeats, args.warmups, args.dtype)
        print(json.dumps(payload))
        return

    sizes_b = [parse_size_to_bytes(s) for s in args.sizes]

    def run_mode(mode: str) -> Dict:
        env = os.environ.copy()

        env["CUPY_ENABLE_UMP"] = "1" if mode == "ump" else "0"
        env["HSA_XNACK"] = "1" if mode == "ump" else ""
        cmd = [sys.executable, __file__, "--_worker", "--_mode", mode,
               "--dtype", args.dtype, "--repeats", str(args.repeats),
               "--warmups", str(args.warmups)]
        for s in args.sizes:
            cmd.extend(["--sizes", s])
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"{mode} run failed with exit code {proc.returncode}")
        return json.loads(proc.stdout.strip())

    print("\n=== Running BASELINE (CUPY_ENABLE_UMP=0) ===")
    base = run_mode("baseline")
    print("\n=== Running UMP (CUPY_ENABLE_UMP=1) ===")
    ump = run_mode("ump")

    
    def tabulate(base, ump):
    
        header = [
            "Size",
            "Mode",
            "HtoD ms",  "HtoD GiB/s",
            "DtoH ms",  "DtoH GiB/s",
            "Add ms",
            "Pipeline ms",
        ]
    
        by_b = {r["size_bytes"]: r for r in base["results"]}
        by_u = {r["size_bytes"]: r for r in ump["results"]}

        rows = []
        f = lambda x: f"{x:.3f}"

        for sz in sorted(by_b.keys()):
            b = by_b[sz]
            u = by_u.get(sz)
            if u is None:
                continue

            size_str = human_bytes(sz)

        
            rows.append([
                size_str,
                "BASE",
                f(b["HtoD_asarray_ms"]), f(b["HtoD_GiBps"]),
                f(b["DtoH_get_ms"]),     f(b["DtoH_GiBps"]),
                f(b["add_inplace_ms"]),
                f(b["pipeline_np_cp_compute_np_ms"]),
            ])

            rows.append([
                size_str,
                "UMP",
                f(u["HtoD_asarray_ms"]), f(u["HtoD_GiBps"]),
                f(u["DtoH_get_ms"]),     f(u["DtoH_GiBps"]),
                f(u["add_inplace_ms"]),
                f(u["pipeline_np_cp_compute_np_ms"]),
         ])
        
        return header, rows


    header, rows = tabulate(base, ump)

    colw = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i,h in enumerate(header)]
    def pr(cols): print(" | ".join(c.ljust(colw[i]) for i,c in enumerate(cols)))
    print("\n=== RESULTS (avg over repeats) ===")
    pr(header); print("-+-".join("-"*w for w in colw))
    for r in rows: pr(r)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"\nSaved CSV: {args.csv}")

    def show_meta(tag, m):
        print(f"[{tag}] device={m['device']['name']}, SMs={m['device']['multiProcessorCount']}, "
              f"mem={human_bytes(m['device']['totalGlobalMem'])}, dtype={m['dtype']}, "
              f"repeats={m['repeats']}, warmups={m['warmups']}")
    show_meta("BASELINE", base["meta"]); show_meta("UMP", ump["meta"])

if __name__ == "__main__":
    main()

