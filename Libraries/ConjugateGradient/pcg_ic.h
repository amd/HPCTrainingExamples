/**
 * Preconditioned Conjugate Gradient with Incomplete Cholesky Preconditioner
 * ROCm 7.1.1 Implementation - Header
 */

#ifndef PCG_IC_H
#define PCG_IC_H

#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>

/**
 * Preconditioned Conjugate Gradient solver with Incomplete Cholesky preconditioner
 * 
 * Solves Ax = b where A is symmetric positive definite sparse matrix in CSR format
 * 
 * @param handle        rocSPARSE handle
 * @param blas_handle   rocBLAS handle
 * @param n             Matrix dimension
 * @param nnz           Number of non-zeros in A
 * @param d_csr_row_ptr CSR row pointers (device)
 * @param d_csr_col_ind CSR column indices (device)
 * @param d_csr_val     CSR values (device)
 * @param d_b           Right-hand side vector (device)
 * @param d_x           Solution vector (device), initial guess on input
 * @param tol           Relative tolerance
 * @param max_iter      Maximum iterations
 * @param iter          Output: actual iterations performed
 * @param final_res     Output: final residual norm
 */
void pcg_incomplete_cholesky(
    rocsparse_handle handle,
    rocblas_handle blas_handle,
    int64_t n,
    int64_t nnz,
    int* d_csr_row_ptr,
    int* d_csr_col_ind,
    double* d_csr_val,
    const double* d_b,
    double* d_x,
    double tol,
    int max_iter,
    int* iter,
    double* final_res);

#endif // PCG_IC_H
