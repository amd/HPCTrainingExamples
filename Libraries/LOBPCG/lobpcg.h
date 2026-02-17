/**
 * LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
 * ROCm 7.1.1 Implementation - Header
 * 
 * Based on lobpcg.m reference implementation
 */

#ifndef LOBPCG_H
#define LOBPCG_H

#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

/**
 * LOBPCG eigenvalue solver
 * 
 * Computes the smallest eigenvalues and corresponding eigenvectors of A*x = lambda*x
 * 
 * @param handle       rocsparse handle
 * @param blas_handle  rocblas handle
 * @param n            Matrix dimension
 * @param nnz          Number of non-zeros in A
 * @param d_csr_row_ptr CSR row pointers (device)
 * @param d_csr_col_ind CSR column indices (device)
 * @param d_csr_val    CSR values (device)
 * @param nev          Number of eigenvalues to compute
 * @param d_X          Eigenvectors output, n x nev column-major (device)
 * @param d_lambda     Eigenvalues output, nev values (device)
 * @param tol          Convergence tolerance for residuals
 * @param max_iter     Maximum number of iterations
 * @param iter         Output: number of iterations performed
 * @param use_precond  Whether to use IC(0) preconditioner
 */
void lobpcg(
    rocsparse_handle handle,
    rocblas_handle blas_handle,
    int64_t n,
    int64_t nnz,
    int* d_csr_row_ptr,
    int* d_csr_col_ind,
    double* d_csr_val,
    int nev,
    double* d_X,
    double* d_lambda,
    double tol,
    int max_iter,
    int* iter,
    bool use_precond);

#endif // LOBPCG_H
