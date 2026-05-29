# Preconditioned Conjugate Gradient (PCG) with Incomplete Cholesky

ROCm implementation of the Preconditioned Conjugate Gradient method with
Incomplete Cholesky (IC0) preconditioner for solving symmetric positive
definite linear systems.

## Requirements

- ROCm 6.4+ or ROCm 7.x
- AMD GPU with ROCm support

## Building

```bash
module load rocm/7.1.1   # or rocm/6.4.0
make
```

This builds two executables:
- `pcg_laplacian` - Test driver with generated 3D Laplacian matrix
- `pcg_mtx` - Driver for Matrix Market format input files

## Usage

### Laplacian Test Driver

```bash
# Default 20x20x20 grid
./pcg_laplacian

# Custom grid size and solver parameters
./pcg_laplacian <nx> <ny> <nz> [tol] [maxiter]

# Example: 50x50x50 grid with tolerance 1e-10
./pcg_laplacian 50 50 50 1e-10 1000
```

### Matrix Market Driver

```bash
./pcg_mtx <matrix.mtx> [options]

Options:
  -tol <value>      Convergence tolerance (default: 1e-10)
  -maxiter <value>  Maximum iterations (default: 10000)
  -rhs <file.mtx>   Right-hand side vector in Matrix Market format
                    (default: vector of all ones)
```

Example:
```bash
./pcg_mtx thermal2.mtx -tol 1e-8 -maxiter 2000
./pcg_mtx matrix.mtx -rhs rhs.mtx -tol 1e-10
```

## Algorithm

The implementation uses:
- **Incomplete Cholesky IC(0)** factorization via `rocsparse_dcsric0`
- **Sparse triangular solves** via `rocsparse_spsv` for preconditioning
- **Sparse matrix-vector product** via `rocsparse_spmv`
- **rocBLAS** for vector operations (dot products, scaling, axpy)

## Files

- `pcg_ic.h` - Header file with function declarations
- `pcg_ic.cpp` - PCG algorithm implementation
- `driver.cpp` - Laplacian test driver
- `driver_mtx.cpp` - Matrix Market driver

## Version Compatibility

The code includes version detection macros to handle API differences
between ROCm 6.4 and 7.x, suppressing deprecation warnings while
maintaining backward compatibility.
