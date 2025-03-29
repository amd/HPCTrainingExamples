# Purpose of this code
Assume you need to solve a linear system

$$
A x = b
$$ 

where $A$ is an $N \times N$ sparse, symmetric positive definite (SPD) matrix, $b$ is an $N\times 1$ right-hand side and $x$ is a vector of unknows. Such a system can be usually solved with a linear solver called [Conjugate Gradient (CG)] (https://en.wikipedia.org/wiki/Conjugate_gradient_method).  

CG is an iterative method that finds an approximate solution to the linear system above. CG implementation consists of sparse matrix-vector products, vector dot products, vector scaling and vector updates (AXPYs). It can be implemented using rocBLAS and rocSPARSE.

This example demonstrates how to:
- use rocBLAS,
- use rocSPARSE,
- create an incomplete Cholesky preconditioner and use it in the code.

# Compiling

The example is self-contained. You only need rocm and gcc.

After cloning, just type `make`.

# Running

```
./run_cg --matrix /path/to/matrix/in/matrix/market/format.mtx --maxit 10000 --tol 1e-8 --rhs /path/to/rhs/in/matrix/market/format.mtx
```

The parameters `tol`, `maxit` and `rhs` are optional.
Matrix MUST BE SPD in order for the code to work.
