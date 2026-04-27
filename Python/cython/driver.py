#!/usr/bin/env python3
"""
Driver for the Cython-wrapped HPC Training Examples kernels.

Exercises every kernel from the shared library and validates
results against NumPy equivalents, matching the behaviour of
the original C programs in the repo.
"""

import time
import numpy as np

try:
    import compute
except ImportError:
    raise ImportError(
        "Could not import the 'compute' shared library.\n"
        "Build it first:  python setup.py build_ext --inplace"
    )


def bench(label, func, *args, repeats=50):
    """Time a function and return its result."""
    result = func(*args)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    elapsed = (time.perf_counter() - t0) / repeats
    print(f"  {label:40s}  {elapsed*1e6:10.1f} us")
    return result


def test_cpu_func():
    """ManagedMemory/CPU_Code/cpu_code.c — doubles every element."""
    M = 100_000
    inp = np.ones(M, dtype=np.float64)

    out = bench("Cython  cpu_func", compute.py_cpu_func, inp)

    # The original C program expects sum(out) == 200000
    assert np.allclose(out, inp * 2.0), "cpu_func mismatch!"
    total = out.sum()
    print(f"  Result is {total:.6f}  (expected {M * 2.0:.6f})")


def test_saxpy():
    """Pragma_Examples/OpenMP/C/1_saxpy — y = a*x + y."""
    N = 1_000_000
    a = np.float32(2.0)
    x = np.ones(N, dtype=np.float32)
    y = np.full(N, 2.0, dtype=np.float32)

    y_out = bench("Cython  saxpy", compute.py_saxpy, a, x, y)
    y_ref = bench("NumPy   a*x + y", lambda: a * x + y)

    # Original program expects y[0] == 4.0, y[N-1] == 4.0
    assert np.allclose(y_out, a * x + y), "saxpy mismatch!"
    print(f"  y[0] {y_out[0]:.6f}  y[N-1] {y_out[-1]:.6f}  (expected 4.0)")


def test_vecadd():
    """Pragma_Examples/OpenMP/C/3_vecadd — c = a + b."""
    N = 100_000
    a = np.array([np.sin(i+1)**2 for i in range(N)], dtype=np.float64)
    b = np.array([np.cos(i+1)**2 for i in range(N)], dtype=np.float64)

    c_cy = bench("Cython  vecadd", compute.py_vecadd, a, b)
    c_np = bench("NumPy   a + b", np.add, a, b)

    assert np.allclose(c_cy, c_np), "vecadd mismatch!"
    # Original expects mean(c) ≈ 1.0 (sin²+cos²=1)
    avg = c_cy.mean()
    print(f"  Final result: {avg:.6f}  (expected ≈1.0)")


def test_reduction():
    """Pragma_Examples/OpenMP/C/2_reduction — sum of array."""
    n = 100_000
    x = np.full(n, 2.0, dtype=np.float64)

    s_cy = bench("Cython  reduction", compute.py_reduction, x)
    s_np = bench("NumPy   np.sum", np.sum, x)

    # Original expects sum == 200000
    assert abs(s_cy - s_np) < 1e-6, "reduction mismatch!"
    print(f"  Sum={s_cy:.6f}  (expected {n * 2.0:.6f})")


def main():
    print("=" * 62)
    print("HPCTrainingExamples — Cython Shared Library Tests")
    print("=" * 62)

    print(f"\n--- cpu_func (ManagedMemory/CPU_Code) ---")
    test_cpu_func()

    print(f"\n--- saxpy (Pragma_Examples/OpenMP/C/1_saxpy) ---")
    test_saxpy()

    print(f"\n--- vecadd (Pragma_Examples/OpenMP/C/3_vecadd) ---")
    test_vecadd()

    print(f"\n--- reduction (Pragma_Examples/OpenMP/C/2_reduction) ---")
    test_reduction()

    print(f"\n{'=' * 62}")
    print("All tests passed.")


if __name__ == "__main__":
    main()
