# LOBPCG Eigenvalue Solver

ROCm implementation of the Locally Optimal Block Preconditioned Conjugate
Gradient (LOBPCG) method for computing the smallest eigenvalues and
eigenvectors of symmetric positive definite matrices.

## Requirements

- ROCm 6.4+ or ROCm 7.x
- AMD GPU with ROCm support

## Building

```bash
module load rocm/7.1.1   # or rocm/6.4.0
make
```

This builds:
- `lobpcg_mtx` - LOBPCG solver for Matrix Market format input files

## Usage

```bash
./lobpcg_mtx <matrix.mtx> [options]

Options:
  -nev <value>      Number of eigenvalues to compute (default: 5)
  -tol <value>      Convergence tolerance (default: 1e-6)
  -maxiter <value>  Maximum iterations (default: 100)
  -x0 <file.mtx>    Initial vectors in Matrix Market format
                    (default: random vectors generated on GPU)
  -seed <value>     Random seed for initial vectors (default: 42)
```

Examples:
```bash
# Compute 5 smallest eigenvalues
./lobpcg_mtx matrix.mtx -nev 5 -tol 1e-8 -maxiter 500

# Compute 10 eigenvalues with custom initial vectors
./lobpcg_mtx matrix.mtx -nev 10 -x0 initial_vectors.mtx
```

## Algorithm

The LOBPCG implementation includes:
- **Block iteration** for computing multiple eigenvalues simultaneously
- **Classical Gram-Schmidt with reorthogonalization (CGS2)** for orthonormalization
- **Rayleigh-Ritz procedure** using `rocsolver_dsygv` for the projected eigenvalue problem
- **Eigenpair locking** for converged eigenpairs
- **Column-by-column SpMV** via `rocsparse_spmv` for sparse matrix-vector products
- **rocRAND** for GPU-native random initial vector generation

## Files

- `lobpcg.h` - Header file with function declarations
- `lobpcg.cpp` - LOBPCG algorithm implementation
- `driver_lobpcg.cpp` - Matrix Market driver

## Performance

Benchmarked on AMD Instinct MI300X with tmt_sym matrix (726K x 726K):
- ~6.5 ms per iteration for 5 eigenvalues
- ~3.9x faster than reference C implementation

## Version Compatibility

The code includes version detection macros to handle API differences
between ROCm 6.4 and 7.x, suppressing deprecation warnings while
maintaining backward compatibility.

## Notes

- The solver computes the **smallest** eigenvalues (ascending order)
- For best convergence, the matrix should be symmetric positive definite
- No preconditioner is currently applied (identity preconditioning)
