# Cython Shared Library — HPCTrainingExamples Kernels

This example wraps **actual C computation kernels from the HPCTrainingExamples repo** into a Python-callable shared library (`.so` / `.pyd`) using Cython.

## Wrapped Kernels

| Function | Original Source | Description |
|---|---|---|
| `py_cpu_func(inp)` | `ManagedMemory/CPU_Code/cpu_code.c` | Doubles every element: `out[i] = in[i] * 2.0` |
| `py_saxpy(a, x, y)` | `Pragma_Examples/OpenMP/C/1_saxpy` | SAXPY: `y = a*x + y` |
| `py_vecadd(a, b)` | `Pragma_Examples/OpenMP/C/3_vecadd` | Vector addition: `c = a + b` |
| `py_reduction(x)` | `Pragma_Examples/OpenMP/C/2_reduction` | Sum-reduction of an array |

The core loops in [hpc_kernels.c](hpc_kernels.c) are extracted directly from the original repo sources (with `main()` and OpenMP timing scaffolding removed). The Cython wrapper in [compute.pyx](compute.pyx) calls into these C functions and handles NumPy array ↔ C pointer conversion.

## Prerequisites

```bash
pip install cython numpy
```

## Build

```bash
# Option 1 – Makefile
make build

# Option 2 – setup.py directly
python setup.py build_ext --inplace
```

This compiles `hpc_kernels.c` + `compute.pyx` into `compute.<platform>.so`.

## Run

```bash
make test
# or
python driver.py
```

The driver benchmarks each Cython-wrapped kernel against its NumPy equivalent and validates correctness using the same expected values as the original C programs.

## Clean

```bash
make clean
```

## File Layout

```
Python/cython/
├── hpc_kernels.h      C declarations for the kernels
├── hpc_kernels.c      C kernel implementations (from repo)
├── compute.pyx        Cython wrapper module
├── setup.py           Build script (setuptools + Cython)
├── driver.py          Benchmark / validation driver
├── Makefile           Build automation
└── README.md
```
