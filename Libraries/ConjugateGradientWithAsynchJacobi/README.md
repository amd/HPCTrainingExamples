# Conjugate Gradient with Preconditioners

Solves sparse symmetric positive definite (SPD) linear systems using the Conjugate Gradient method with various preconditioners. Implemented using HIP, rocBLAS, rocSPARSE, and rocSOLVER.

## Executables

- `run_cg` - Standard Preconditioned Conjugate Gradient (PCG)
- `run_cg_cg` - CG-CG variant with optional low-synchronization mode

## Available Preconditioners

| Name | Description |
|------|-------------|
| `none` | No preconditioning |
| `ic` / `ichol` | Incomplete Cholesky |
| `jacobi` | Jacobi (damped Richardson) |
| `asynch_jacobi` | Asynchronous Jacobi |
| `gs_it` | Iterative Gauss-Seidel |
| `gs_it2` | Iterative Gauss-Seidel (variant 2) |

## Compiling

```bash
module load rocm
make
```

To clean and rebuild:

```bash
make clean
make
```

Note: The Makefile uses `--offload-arch=gfx950` by default. Modify `CXXFLAGS` in `Makefile` if targeting a different GPU architecture.

## Usage

### run_cg

```bash
./run_cg --matrix <path.mtx> [options]
```

### run_cg_cg

```bash
./run_cg_cg --matrix <path.mtx> [options]
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--matrix <file>` | Path to matrix in Matrix Market (.mtx) format (required) |
| `--rhs <file>` | Path to right-hand side vector in Matrix Market format |
| `--tol <value>` | Convergence tolerance (default: 1e-8) |
| `--maxit <n>` | Maximum iterations (default: 100000) |
| `--precond <name>` | Preconditioner name (see table above) |
| `--jacobi_iter <n>` | Number of Jacobi iterations |
| `--jacobi_omega <value>` | Jacobi relaxation parameter (0 < omega <= 1) |
| `--asynch_jacobi_version <n>` | Asynchronous Jacobi kernel version |
| `--gs_inner_iter <n>` | Gauss-Seidel inner iterations |
| `--gs_outer_iter <n>` | Gauss-Seidel outer iterations |
| `--low-synch <0\|1>` | Enable low-synchronization mode (`run_cg_cg` only) |

## Example

Download a test matrix:

```bash
wget https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
tar -xvf 1138_bus.tar.gz
```

Run with Jacobi preconditioner:

```bash
./run_cg --matrix 1138_bus/1138_bus.mtx --precond jacobi --jacobi_iter 5 --maxit 10000
```

Run with asynchronous Jacobi:

```bash
./run_cg --matrix 1138_bus/1138_bus.mtx --precond asynch_jacobi --jacobi_iter 3
```

Run CG-CG with low-synch mode:

```bash
./run_cg_cg --matrix 1138_bus/1138_bus.mtx --precond jacobi --low-synch 1
```

## Notes

- Matrix must be in Matrix Market (.mtx) format
- Matrix must be symmetric positive definite (SPD)
- Incomplete Cholesky may fail for some matrices
