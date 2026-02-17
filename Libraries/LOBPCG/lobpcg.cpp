/**
 * LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
 * ROCm Implementation - Algorithm
 * 
 * Compatible with ROCm 6.4+ and ROCm 7.x
 * 
 * Based on lobpcg.m reference implementation
 * Computes smallest eigenvalues of symmetric positive definite matrices
 */

#include "lobpcg.h"
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse-version.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

// Version compatibility macros
// ROCm 7.x (rocsparse 4.x) deprecates rocsparse_spmv in favor of rocsparse_v2_spmv
// but rocsparse_spmv is still available for compatibility
#define ROCSPARSE_VERSION_CODE(major, minor) ((major) * 100 + (minor))
#define ROCSPARSE_VERSION_CURRENT ROCSPARSE_VERSION_CODE(ROCSPARSE_VERSION_MAJOR, ROCSPARSE_VERSION_MINOR)
#define ROCSPARSE_VERSION_6_4 ROCSPARSE_VERSION_CODE(3, 4)  // ROCm 6.4 = rocsparse 3.4
#define ROCSPARSE_VERSION_7_0 ROCSPARSE_VERSION_CODE(4, 0)  // ROCm 7.x = rocsparse 4.x

// Suppress deprecation warnings for rocsparse_spmv in ROCm 7.x
// We use the old API for compatibility with ROCm 6.4
#if ROCSPARSE_VERSION_CURRENT >= ROCSPARSE_VERSION_7_0
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error: " << hipGetErrorString(err)                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define ROCSPARSE_CHECK(call)                                                   \
    do {                                                                        \
        rocsparse_status status = call;                                         \
        if (status != rocsparse_status_success) {                               \
            std::cerr << "rocSPARSE error: " << status                          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define ROCBLAS_CHECK(call)                                                     \
    do {                                                                        \
        rocblas_status status = call;                                           \
        if (status != rocblas_status_success) {                                 \
            std::cerr << "rocBLAS error: " << status                            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/**
 * Compute column norms of a matrix
 * d_V: n x k column-major matrix
 * d_norms: output array of k norms
 */
__global__ void compute_column_norms_kernel(const double* d_V,
                                            int64_t n,
                                            int k,
                                            double* d_norms)
{
    int col = blockIdx.x;
    if (col >= k) return;
    
    __shared__ double partial_sum[256];
    
    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        double val = d_V[col * n + i];
        sum += val * val;
    }
    
    partial_sum[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        d_norms[col] = sqrt(partial_sum[0]);
    }
}

/**
 * Normalize columns of a matrix
 */
__global__ void normalize_columns_kernel(double* d_V,
                                         int64_t n,
                                         int k,
                                         const double* d_norms)
{
    int col = blockIdx.x;
    if (col >= k) return;
    
    double norm = d_norms[col];
    if (norm < 1e-14) norm = 1.0;  // Avoid division by zero
    
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        d_V[col * n + i] /= norm;
    }
}

/**
 * CGS2 workspace structure (pre-allocated for efficiency)
 */
struct cgs2_workspace {
    int k_max;      // Maximum number of columns supported
    double* d_a1;   // Device buffer for first orthogonalization coefficients
    double* d_a2;   // Device buffer for second orthogonalization coefficients
};

/**
 * Allocate CGS2 workspace
 */
cgs2_workspace* cgs2_workspace_alloc(int k_max)
{
    cgs2_workspace* ws = new cgs2_workspace;
    ws->k_max = k_max;
    HIP_CHECK(hipMalloc(&ws->d_a1, k_max * sizeof(double)));
    HIP_CHECK(hipMalloc(&ws->d_a2, k_max * sizeof(double)));
    return ws;
}

/**
 * Free CGS2 workspace
 */
void cgs2_workspace_free(cgs2_workspace* ws)
{
    if (ws == nullptr) return;
    HIP_CHECK(hipFree(ws->d_a1));
    HIP_CHECK(hipFree(ws->d_a2));
    delete ws;
}

/**
 * CGS2 (Classical Gram-Schmidt with reorthogonalization)
 * Orthonormalizes columns of V in-place
 * d_V: n x k column-major matrix (modified in place)
 * ws: pre-allocated workspace (must have k_max >= k)
 */
void cgs2_with_workspace(rocblas_handle blas_handle,
                         double* d_V,
                         int64_t n,
                         int k,
                         cgs2_workspace* ws)
{
    const double one = 1.0;
    const double neg_one = -1.0;
    const double zero = 0.0;
    
    double* d_a1 = ws->d_a1;
    double* d_a2 = ws->d_a2;
    
    for (int i = 0; i < k; i++) {
        double* col_i = d_V + i * n;
        
        if (i > 0) {
            // First orthogonalization pass
            // a1 = V(:,0:i-1)' * V(:,i)
            ROCBLAS_CHECK(rocblas_dgemv(blas_handle,
                                        rocblas_operation_transpose,
                                        n,
                                        i,
                                        &one,
                                        d_V,
                                        n,
                                        col_i,
                                        1,
                                        &zero,
                                        d_a1,
                                        1));
            
            // V(:,i) = V(:,i) - V(:,0:i-1) * a1
            ROCBLAS_CHECK(rocblas_dgemv(blas_handle,
                                        rocblas_operation_none,
                                        n,
                                        i,
                                        &neg_one,
                                        d_V,
                                        n,
                                        d_a1,
                                        1,
                                        &one,
                                        col_i,
                                        1));
            
            // Second orthogonalization pass (reorthogonalization)
            // a2 = V(:,0:i-1)' * V(:,i)
            ROCBLAS_CHECK(rocblas_dgemv(blas_handle,
                                        rocblas_operation_transpose,
                                        n,
                                        i,
                                        &one,
                                        d_V,
                                        n,
                                        col_i,
                                        1,
                                        &zero,
                                        d_a2,
                                        1));
            
            // V(:,i) = V(:,i) - V(:,0:i-1) * a2
            ROCBLAS_CHECK(rocblas_dgemv(blas_handle,
                                        rocblas_operation_none,
                                        n,
                                        i,
                                        &neg_one,
                                        d_V,
                                        n,
                                        d_a2,
                                        1,
                                        &one,
                                        col_i,
                                        1));
        }
        
        // Normalize column i
        double norm;
        ROCBLAS_CHECK(rocblas_dnrm2(blas_handle,
                                    n,
                                    col_i,
                                    1,
                                    &norm));
        
        if (norm > 1e-14) {
            double inv_norm = 1.0 / norm;
            ROCBLAS_CHECK(rocblas_dscal(blas_handle,
                                        n,
                                        &inv_norm,
                                        col_i,
                                        1));
        }
    }
}

/**
 * Sparse matrix - dense matrix multiplication using column-by-column SpMV
 * C = A * B where A is sparse (n x n), B is dense (n x k), C is dense (n x k)
 * This matches the approach in the reference implementation (csr_matmat)
 */
void csr_matmat(rocsparse_handle handle,
                rocsparse_spmat_descr mat_A,
                const double* d_B,
                double* d_C,
                int64_t n,
                int k,
                rocsparse_dnvec_descr* vec_in,
                rocsparse_dnvec_descr* vec_out,
                void* d_buffer,
                size_t buffer_size)
{
    const double one = 1.0;
    const double zero = 0.0;
    
    // Multiply column by column (like reference implementation)
    for (int j = 0; j < k; ++j) {
        const double* bj = d_B + j * n;
        double* cj = d_C + j * n;
        
        // Update vector descriptors to point to current columns
        ROCSPARSE_CHECK(rocsparse_dnvec_set_values(*vec_in, (void*)bj));
        ROCSPARSE_CHECK(rocsparse_dnvec_set_values(*vec_out, (void*)cj));
        
        ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                       rocsparse_operation_none,
                                       &one,
                                       mat_A,
                                       *vec_in,
                                       &zero,
                                       *vec_out,
                                       rocsparse_datatype_f64_r,
                                       rocsparse_spmv_alg_default,
                                       rocsparse_spmv_stage_compute,
                                       &buffer_size,
                                       d_buffer));
    }
}

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
    bool use_precond)
{
    const double one = 1.0;
    const double neg_one = -1.0;
    const double zero = 0.0;
    
    int k = nev;  // Number of eigenvalues to compute
    
    // Create sparse matrix descriptor for A
    rocsparse_spmat_descr mat_A;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&mat_A,
                                               n,
                                               n,
                                               nnz,
                                               d_csr_row_ptr,
                                               d_csr_col_ind,
                                               d_csr_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f64_r));
    
    // Allocate temporary vectors for SpMV (column-by-column approach)
    double* d_spmv_tmp;
    HIP_CHECK(hipMalloc(&d_spmv_tmp, n * sizeof(double)));
    
    // Create vector descriptors for SpMV
    rocsparse_dnvec_descr vec_spmv_in, vec_spmv_out;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_spmv_in,
                                                  n,
                                                  d_spmv_tmp,
                                                  rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_spmv_out,
                                                  n,
                                                  d_spmv_tmp,
                                                  rocsparse_datatype_f64_r));
    
    // Get SpMV buffer size
    size_t spmv_buffer_size;
    ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                   rocsparse_operation_none,
                                   &one,
                                   mat_A,
                                   vec_spmv_in,
                                   &zero,
                                   vec_spmv_out,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spmv_alg_default,
                                   rocsparse_spmv_stage_buffer_size,
                                   &spmv_buffer_size,
                                   nullptr));
    
    void* d_spmv_buffer;
    HIP_CHECK(hipMalloc(&d_spmv_buffer, spmv_buffer_size));
    
    // Allocate working arrays
    // Maximum subspace size is 3*k (X, W, P)
    int max_subspace = 3 * k;
    
    double* d_AX;      // n x k
    double* d_R;       // n x k (residuals)
    double* d_W;       // n x k (preconditioned residuals)
    double* d_P;       // n x k (search directions)
    double* d_AP;      // n x k
    double* d_AW;      // n x k
    double* d_S;       // n x max_subspace (trial subspace)
    double* d_AS;      // n x max_subspace
    double* d_Lambda;  // k x k (Rayleigh quotient)
    
    HIP_CHECK(hipMalloc(&d_AX, n * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_R, n * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W, n * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_P, n * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_AP, n * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_AW, n * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_S, n * max_subspace * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_AS, n * max_subspace * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Lambda, k * k * sizeof(double)));
    
    // CGS2 workspace (allocated once, reused for all cgs2 calls)
    cgs2_workspace* cgs2_ws = cgs2_workspace_alloc(max_subspace);
    
    // For small dense eigenvalue problem
    double* d_AS_proj;  // max_subspace x max_subspace
    double* d_BS_proj;  // max_subspace x max_subspace
    double* d_Y;        // eigenvectors of projected problem
    double* d_theta;    // eigenvalues of projected problem
    double* d_E;        // off-diagonal elements for rocsolver
    
    HIP_CHECK(hipMalloc(&d_AS_proj, max_subspace * max_subspace * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_BS_proj, max_subspace * max_subspace * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Y, max_subspace * max_subspace * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_theta, max_subspace * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_E, max_subspace * sizeof(double)));
    
    // rocsolver workspace for dsygv
    rocblas_int* d_info;
    HIP_CHECK(hipMalloc(&d_info, sizeof(rocblas_int)));
    
    // Host arrays for intermediate results
    std::vector<double> h_norms(k);
    std::vector<double> h_theta(max_subspace);
    std::vector<double> h_Y(max_subspace * max_subspace);
    std::vector<double> h_AS_proj(max_subspace * max_subspace);
    std::vector<double> h_BS_proj(max_subspace * max_subspace);
    
    // Locking: storage for converged eigenpairs
    double* d_X_lock;
    HIP_CHECK(hipMalloc(&d_X_lock, n * k * sizeof(double)));
    std::vector<double> h_lambda_lock(k);
    int n_locked = 0;
    
    // Workspace for orthogonalization against locked vectors
    double* d_lock_coeff;
    HIP_CHECK(hipMalloc(&d_lock_coeff, k * k * sizeof(double)));
    
    // Initialize P to zero (no search direction initially)
    HIP_CHECK(hipMemset(d_P, 0, n * k * sizeof(double)));
    HIP_CHECK(hipMemset(d_AP, 0, n * k * sizeof(double)));
    
    bool has_P = false;
    int k_active = k;
    
    // Initial orthonormalization of X
    cgs2_with_workspace(blas_handle, d_X, n, k, cgs2_ws);
    
    // Main iteration loop
    for (int it = 1; it <= max_iter; it++) {
        // Compute AX = A * X (column-by-column SpMV)
        csr_matmat(handle, mat_A, d_X, d_AX, n, k_active, 
                   &vec_spmv_in, &vec_spmv_out, d_spmv_buffer, spmv_buffer_size);
        
        // Compute Rayleigh quotient Lambda = X' * AX
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_transpose,
                                    rocblas_operation_none,
                                    k_active,
                                    k_active,
                                    n,
                                    &one,
                                    d_X,
                                    n,
                                    d_AX,
                                    n,
                                    &zero,
                                    d_Lambda,
                                    k_active));
        
        // Compute residuals R = AX - X * Lambda
        HIP_CHECK(hipMemcpy(d_R,
                            d_AX,
                            n * k_active * sizeof(double),
                            hipMemcpyDeviceToDevice));
        
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    n,
                                    k_active,
                                    k_active,
                                    &neg_one,
                                    d_X,
                                    n,
                                    d_Lambda,
                                    k_active,
                                    &one,
                                    d_R,
                                    n));
        
        // Compute residual norms
        double max_res = 0.0;
        for (int i = 0; i < k_active; i++) {
            double norm;
            ROCBLAS_CHECK(rocblas_dnrm2(blas_handle,
                                        n,
                                        d_R + i * n,
                                        1,
                                        &norm));
            h_norms[i] = norm;
            if (norm > max_res) max_res = norm;
        }
        
        // Copy eigenvalues to host for printing
        std::vector<double> h_lambda(k_active);
        HIP_CHECK(hipMemcpy(h_lambda.data(),
                            d_Lambda,
                            k_active * sizeof(double),
                            hipMemcpyDeviceToHost));  // diagonal elements
        
        // Extract diagonal (Lambda is k_active x k_active)
        std::vector<double> h_Lambda_full(k_active * k_active);
        HIP_CHECK(hipMemcpy(h_Lambda_full.data(),
                            d_Lambda,
                            k_active * k_active * sizeof(double),
                            hipMemcpyDeviceToHost));
        for (int i = 0; i < k_active; i++) {
            h_lambda[i] = h_Lambda_full[i * k_active + i];  // diagonal
        }
        
        printf("it %4d  max residual = %.3e\n", it, max_res);
        for (int i = 0; i < k_active; i++) {
            printf("  eigenvalue %d: %15.10e  residual: %.3e\n", 
                   i + 1, h_lambda[i], h_norms[i]);
        }
        
        // ------------------------------ 
        // Lock converged eigenpairs
        // ------------------------------ 
        std::vector<int> locked(k_active, 0);
        int any_locked = 0;
        for (int i = 0; i < k_active; i++) {
            if (h_norms[i] < tol) {
                locked[i] = 1;
                any_locked = 1;
            }
        }
        
        if (any_locked) {
            // Copy locked eigenpairs to X_lock and lambda_lock
            for (int i = 0; i < k_active; i++) {
                if (locked[i]) {
                    HIP_CHECK(hipMemcpy(d_X_lock + n_locked * n,
                                        d_X + i * n,
                                        n * sizeof(double),
                                        hipMemcpyDeviceToDevice));
                    h_lambda_lock[n_locked] = h_lambda[i];
                    printf("  Eigenpair %d converged, lambda = %.10e, res = %.3e\n",
                           n_locked + 1, h_lambda[i], h_norms[i]);
                    n_locked++;
                }
            }
            
            // Check if all converged
            if (n_locked >= k) {
                printf("LOBPCG: All %d eigenpairs converged\n", k);
                *iter = it;
                // Copy locked eigenvalues to output
                for (int i = 0; i < k; i++) {
                    HIP_CHECK(hipMemcpy(d_lambda + i,
                                        &h_lambda_lock[i],
                                        sizeof(double),
                                        hipMemcpyHostToDevice));
                }
                // Copy locked eigenvectors to output
                HIP_CHECK(hipMemcpy(d_X,
                                    d_X_lock,
                                    n * k * sizeof(double),
                                    hipMemcpyDeviceToDevice));
                goto cleanup;
            }
            
            // Compact X, R, P to remove locked columns
            int write_idx = 0;
            for (int i = 0; i < k_active; i++) {
                if (!locked[i]) {
                    if (write_idx != i) {
                        HIP_CHECK(hipMemcpy(d_X + write_idx * n,
                                            d_X + i * n,
                                            n * sizeof(double),
                                            hipMemcpyDeviceToDevice));
                        HIP_CHECK(hipMemcpy(d_R + write_idx * n,
                                            d_R + i * n,
                                            n * sizeof(double),
                                            hipMemcpyDeviceToDevice));
                        if (has_P) {
                            HIP_CHECK(hipMemcpy(d_P + write_idx * n,
                                                d_P + i * n,
                                                n * sizeof(double),
                                                hipMemcpyDeviceToDevice));
                        }
                        h_lambda[write_idx] = h_lambda[i];
                    }
                    write_idx++;
                }
            }
            k_active = write_idx;
            
            if (k_active <= 0) {
                printf("LOBPCG: All eigenpairs converged after compaction\n");
                *iter = it;
                for (int i = 0; i < k; i++) {
                    HIP_CHECK(hipMemcpy(d_lambda + i,
                                        &h_lambda_lock[i],
                                        sizeof(double),
                                        hipMemcpyHostToDevice));
                }
                HIP_CHECK(hipMemcpy(d_X,
                                    d_X_lock,
                                    n * k * sizeof(double),
                                    hipMemcpyDeviceToDevice));
                goto cleanup;
            }
        }
        
        // ------------------------------ 
        // Apply preconditioner: W = prec(R)
        // For now, just use identity (W = R)
        // ------------------------------ 
        HIP_CHECK(hipMemcpy(d_W,
                            d_R,
                            n * k_active * sizeof(double),
                            hipMemcpyDeviceToDevice));
        
        // ------------------------------ 
        // Orthogonalize W and P against locked vectors
        // ------------------------------ 
        if (n_locked > 0) {
            // W = W - X_lock * (X_lock' * W)
            ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                        rocblas_operation_transpose,
                                        rocblas_operation_none,
                                        n_locked,
                                        k_active,
                                        n,
                                        &one,
                                        d_X_lock,
                                        n,
                                        d_W,
                                        n,
                                        &zero,
                                        d_lock_coeff,
                                        n_locked));
            ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                        rocblas_operation_none,
                                        rocblas_operation_none,
                                        n,
                                        k_active,
                                        n_locked,
                                        &neg_one,
                                        d_X_lock,
                                        n,
                                        d_lock_coeff,
                                        n_locked,
                                        &one,
                                        d_W,
                                        n));
            
            // P = P - X_lock * (X_lock' * P)
            if (has_P) {
                ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                            rocblas_operation_transpose,
                                            rocblas_operation_none,
                                            n_locked,
                                            k_active,
                                            n,
                                            &one,
                                            d_X_lock,
                                            n,
                                            d_P,
                                            n,
                                            &zero,
                                            d_lock_coeff,
                                            n_locked));
                ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                            rocblas_operation_none,
                                            rocblas_operation_none,
                                            n,
                                            k_active,
                                            n_locked,
                                            &neg_one,
                                            d_X_lock,
                                            n,
                                            d_lock_coeff,
                                            n_locked,
                                            &one,
                                            d_P,
                                            n));
            }
        }
        
        // ------------------------------ 
        // Orthogonalize W against X
        // W = W - X * (X' * W)
        // ------------------------------ 
        double* d_XtW;
        HIP_CHECK(hipMalloc(&d_XtW, k_active * k_active * sizeof(double)));
        
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_transpose,
                                    rocblas_operation_none,
                                    k_active,
                                    k_active,
                                    n,
                                    &one,
                                    d_X,
                                    n,
                                    d_W,
                                    n,
                                    &zero,
                                    d_XtW,
                                    k_active));
        
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    n,
                                    k_active,
                                    k_active,
                                    &neg_one,
                                    d_X,
                                    n,
                                    d_XtW,
                                    k_active,
                                    &one,
                                    d_W,
                                    n));
        
        // Orthogonalize W against P (if exists)
        if (has_P) {
            ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                        rocblas_operation_transpose,
                                        rocblas_operation_none,
                                        k_active,
                                        k_active,
                                        n,
                                        &one,
                                        d_P,
                                        n,
                                        d_W,
                                        n,
                                        &zero,
                                        d_XtW,
                                        k_active));
            
            ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                        rocblas_operation_none,
                                        rocblas_operation_none,
                                        n,
                                        k_active,
                                        k_active,
                                        &neg_one,
                                        d_P,
                                        n,
                                        d_XtW,
                                        k_active,
                                        &one,
                                        d_W,
                                        n));
        }
        
        HIP_CHECK(hipFree(d_XtW));
        
        // Drop tiny W directions (like MATLAB: idxW = vecnorm(W) > 1e-12)
        int kW = 0;
        for (int i = 0; i < k_active; i++) {
            double norm;
            ROCBLAS_CHECK(rocblas_dnrm2(blas_handle,
                                        n,
                                        d_W + i * n,
                                        1,
                                        &norm));
            if (norm > 1e-12) {
                if (kW != i) {
                    HIP_CHECK(hipMemcpy(d_W + kW * n,
                                        d_W + i * n,
                                        n * sizeof(double),
                                        hipMemcpyDeviceToDevice));
                }
                kW++;
            }
        }
        
        if (kW == 0) {
            printf("Warning: All W directions collapsed, stopping.\n");
            *iter = it;
            for (int i = 0; i < k_active; i++) {
                HIP_CHECK(hipMemcpy(d_lambda + i,
                                    &h_lambda[i],
                                    sizeof(double),
                                    hipMemcpyHostToDevice));
            }
            goto cleanup;
        }
        
        // Orthonormalize W using CGS2 (like reference implementation)
        cgs2_with_workspace(blas_handle, d_W, n, kW, cgs2_ws);
        
        // Count valid P directions
        int kP = 0;
        if (has_P) {
            for (int i = 0; i < k_active; i++) {
                double norm;
                ROCBLAS_CHECK(rocblas_dnrm2(blas_handle,
                                            n,
                                            d_P + i * n,
                                            1,
                                            &norm));
                if (norm > 1e-12) {
                    if (kP != i) {
                        HIP_CHECK(hipMemcpy(d_P + kP * n,
                                            d_P + i * n,
                                            n * sizeof(double),
                                            hipMemcpyDeviceToDevice));
                    }
                    kP++;
                }
            }
        }
        
        // Build trial subspace S = [X, W, P]
        int kX = k_active;
        int subspace_size = kX + kW + kP;
        
        // Copy X to S
        HIP_CHECK(hipMemcpy(d_S,
                            d_X,
                            n * kX * sizeof(double),
                            hipMemcpyDeviceToDevice));
        
        // Copy W to S
        HIP_CHECK(hipMemcpy(d_S + kX * n,
                            d_W,
                            n * kW * sizeof(double),
                            hipMemcpyDeviceToDevice));
        
        // Copy P to S (if exists)
        if (kP > 0) {
            HIP_CHECK(hipMemcpy(d_S + (kX + kW) * n,
                                d_P,
                                n * kP * sizeof(double),
                                hipMemcpyDeviceToDevice));
        }
        
        // Compute AS = A * S (column-by-column SpMV)
        csr_matmat(handle, mat_A, d_S, d_AS, n, subspace_size,
                   &vec_spmv_in, &vec_spmv_out, d_spmv_buffer, spmv_buffer_size);
        
        // Compute projected matrices
        // AS_proj = S' * AS
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_transpose,
                                    rocblas_operation_none,
                                    subspace_size,
                                    subspace_size,
                                    n,
                                    &one,
                                    d_S,
                                    n,
                                    d_AS,
                                    n,
                                    &zero,
                                    d_AS_proj,
                                    subspace_size));
        
        // BS_proj = S' * S
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_transpose,
                                    rocblas_operation_none,
                                    subspace_size,
                                    subspace_size,
                                    n,
                                    &one,
                                    d_S,
                                    n,
                                    d_S,
                                    n,
                                    &zero,
                                    d_BS_proj,
                                    subspace_size));
        
        // Symmetrize (for numerical stability)
        HIP_CHECK(hipMemcpy(h_AS_proj.data(),
                            d_AS_proj,
                            subspace_size * subspace_size * sizeof(double),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_BS_proj.data(),
                            d_BS_proj,
                            subspace_size * subspace_size * sizeof(double),
                            hipMemcpyDeviceToHost));
        
        for (int i = 0; i < subspace_size; i++) {
            for (int j = i + 1; j < subspace_size; j++) {
                double avg_A = 0.5 * (h_AS_proj[i * subspace_size + j] + 
                                      h_AS_proj[j * subspace_size + i]);
                h_AS_proj[i * subspace_size + j] = avg_A;
                h_AS_proj[j * subspace_size + i] = avg_A;
                
                double avg_B = 0.5 * (h_BS_proj[i * subspace_size + j] + 
                                      h_BS_proj[j * subspace_size + i]);
                h_BS_proj[i * subspace_size + j] = avg_B;
                h_BS_proj[j * subspace_size + i] = avg_B;
            }
        }
        
        HIP_CHECK(hipMemcpy(d_AS_proj,
                            h_AS_proj.data(),
                            subspace_size * subspace_size * sizeof(double),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_BS_proj,
                            h_BS_proj.data(),
                            subspace_size * subspace_size * sizeof(double),
                            hipMemcpyHostToDevice));
        
        // Solve generalized eigenvalue problem: AS_proj * Y = BS_proj * Y * D
        // Using rocsolver_dsygv (type 1: A*x = lambda*B*x)
        ROCBLAS_CHECK(rocsolver_dsygv(blas_handle,
                                      rocblas_eform_ax,
                                      rocblas_evect_original,
                                      rocblas_fill_upper,
                                      subspace_size,
                                      d_AS_proj,
                                      subspace_size,
                                      d_BS_proj,
                                      subspace_size,
                                      d_theta,
                                      d_E,
                                      d_info));
        
        // Check for errors
        rocblas_int h_info;
        HIP_CHECK(hipMemcpy(&h_info,
                            d_info,
                            sizeof(rocblas_int),
                            hipMemcpyDeviceToHost));
        if (h_info != 0) {
            printf("Warning: rocsolver_dsygv returned info = %d\n", h_info);
        }
        
        // Eigenvalues are in d_theta, eigenvectors overwrite d_AS_proj
        // Sort by ascending eigenvalue and take first k_active
        HIP_CHECK(hipMemcpy(h_theta.data(),
                            d_theta,
                            subspace_size * sizeof(double),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_Y.data(),
                            d_AS_proj,
                            subspace_size * subspace_size * sizeof(double),
                            hipMemcpyDeviceToHost));
        
        // Create index array for sorting
        std::vector<int> idx(subspace_size);
        for (int i = 0; i < subspace_size; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&h_theta](int a, int b) {
            return h_theta[a] < h_theta[b];
        });
        
        // Extract first k_active eigenvectors (sorted)
        std::vector<double> h_Y_sorted(subspace_size * k_active);
        for (int j = 0; j < k_active; j++) {
            for (int i = 0; i < subspace_size; i++) {
                h_Y_sorted[j * subspace_size + i] = h_Y[idx[j] * subspace_size + i];
            }
        }
        
        HIP_CHECK(hipMemcpy(d_Y,
                            h_Y_sorted.data(),
                            subspace_size * k_active * sizeof(double),
                            hipMemcpyHostToDevice));
        
        // Partition Y into Yx, Yw, Yp
        // Y is subspace_size x k_active
        // Yx = Y(0:kX-1, :)
        // Yw = Y(kX:kX+kW-1, :)
        // Yp = Y(kX+kW:end, :)
        
        // Compute new X = X * Yx + W * Yw + P * Yp
        // First, extract submatrices of Y
        double* d_Yx;  // kX x k_active
        double* d_Yw;  // kW x k_active
        double* d_Yp;  // kP x k_active
        
        HIP_CHECK(hipMalloc(&d_Yx, kX * k_active * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_Yw, kW * k_active * sizeof(double)));
        if (kP > 0) {
            HIP_CHECK(hipMalloc(&d_Yp, kP * k_active * sizeof(double)));
        }
        
        // Extract Yx (rows 0 to kX-1)
        for (int j = 0; j < k_active; j++) {
            HIP_CHECK(hipMemcpy(d_Yx + j * kX,
                                d_Y + j * subspace_size,
                                kX * sizeof(double),
                                hipMemcpyDeviceToDevice));
        }
        
        // Extract Yw (rows kX to kX+kW-1)
        for (int j = 0; j < k_active; j++) {
            HIP_CHECK(hipMemcpy(d_Yw + j * kW,
                                d_Y + j * subspace_size + kX,
                                kW * sizeof(double),
                                hipMemcpyDeviceToDevice));
        }
        
        // Extract Yp (rows kX+kW to end)
        if (kP > 0) {
            for (int j = 0; j < k_active; j++) {
                HIP_CHECK(hipMemcpy(d_Yp + j * kP,
                                    d_Y + j * subspace_size + kX + kW,
                                    kP * sizeof(double),
                                    hipMemcpyDeviceToDevice));
            }
        }
        
        // Compute Xnew = X * Yx + W * Yw + P * Yp
        double* d_Xnew;
        HIP_CHECK(hipMalloc(&d_Xnew, n * k_active * sizeof(double)));
        
        // Xnew = X * Yx
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    n,
                                    k_active,
                                    kX,
                                    &one,
                                    d_X,
                                    n,
                                    d_Yx,
                                    kX,
                                    &zero,
                                    d_Xnew,
                                    n));
        
        // Xnew += W * Yw
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    n,
                                    k_active,
                                    kW,
                                    &one,
                                    d_W,
                                    n,
                                    d_Yw,
                                    kW,
                                    &one,
                                    d_Xnew,
                                    n));
        
        // Xnew += P * Yp (if P exists)
        if (kP > 0) {
            ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                        rocblas_operation_none,
                                        rocblas_operation_none,
                                        n,
                                        k_active,
                                        kP,
                                        &one,
                                        d_P,
                                        n,
                                        d_Yp,
                                        kP,
                                        &one,
                                        d_Xnew,
                                        n));
        }
        
        // Orthonormalize Xnew
        cgs2_with_workspace(blas_handle, d_Xnew, n, k_active, cgs2_ws);
        
        // Compute Pnew = W * Yw + P * Yp
        double* d_Pnew;
        HIP_CHECK(hipMalloc(&d_Pnew, n * k_active * sizeof(double)));
        
        // Pnew = W * Yw
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    n,
                                    k_active,
                                    kW,
                                    &one,
                                    d_W,
                                    n,
                                    d_Yw,
                                    kW,
                                    &zero,
                                    d_Pnew,
                                    n));
        
        // Pnew += P * Yp (if P exists)
        if (kP > 0) {
            ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                        rocblas_operation_none,
                                        rocblas_operation_none,
                                        n,
                                        k_active,
                                        kP,
                                        &one,
                                        d_P,
                                        n,
                                        d_Yp,
                                        kP,
                                        &one,
                                        d_Pnew,
                                        n));
        }
        
        // Stabilize P: Pnew = Pnew - Xnew * (Xnew' * Pnew)
        double* d_XtP;
        HIP_CHECK(hipMalloc(&d_XtP, k_active * k_active * sizeof(double)));
        
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_transpose,
                                    rocblas_operation_none,
                                    k_active,
                                    k_active,
                                    n,
                                    &one,
                                    d_Xnew,
                                    n,
                                    d_Pnew,
                                    n,
                                    &zero,
                                    d_XtP,
                                    k_active));
        
        ROCBLAS_CHECK(rocblas_dgemm(blas_handle,
                                    rocblas_operation_none,
                                    rocblas_operation_none,
                                    n,
                                    k_active,
                                    k_active,
                                    &neg_one,
                                    d_Xnew,
                                    n,
                                    d_XtP,
                                    k_active,
                                    &one,
                                    d_Pnew,
                                    n));
        
        HIP_CHECK(hipFree(d_XtP));
        
        // Orthonormalize Pnew
        cgs2_with_workspace(blas_handle, d_Pnew, n, k_active, cgs2_ws);
        
        // Update X and P
        HIP_CHECK(hipMemcpy(d_X,
                            d_Xnew,
                            n * k_active * sizeof(double),
                            hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(d_P,
                            d_Pnew,
                            n * k_active * sizeof(double),
                            hipMemcpyDeviceToDevice));
        
        has_P = true;
        
        // Cleanup iteration temporaries
        HIP_CHECK(hipFree(d_Yx));
        HIP_CHECK(hipFree(d_Yw));
        if (kP > 0) {
            HIP_CHECK(hipFree(d_Yp));
        }
        HIP_CHECK(hipFree(d_Xnew));
        HIP_CHECK(hipFree(d_Pnew));
    }
    
    *iter = max_iter;
    
cleanup:
    // Free memory
    HIP_CHECK(hipFree(d_AX));
    HIP_CHECK(hipFree(d_R));
    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_P));
    HIP_CHECK(hipFree(d_AP));
    HIP_CHECK(hipFree(d_AW));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipFree(d_AS));
    HIP_CHECK(hipFree(d_Lambda));
    HIP_CHECK(hipFree(d_X_lock));
    HIP_CHECK(hipFree(d_lock_coeff));
    cgs2_workspace_free(cgs2_ws);
    HIP_CHECK(hipFree(d_AS_proj));
    HIP_CHECK(hipFree(d_BS_proj));
    HIP_CHECK(hipFree(d_Y));
    HIP_CHECK(hipFree(d_theta));
    HIP_CHECK(hipFree(d_E));
    HIP_CHECK(hipFree(d_info));
    HIP_CHECK(hipFree(d_spmv_buffer));
    HIP_CHECK(hipFree(d_spmv_tmp));
    
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_spmv_in));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_spmv_out));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(mat_A));
}

// Restore warning settings
#if ROCSPARSE_VERSION_CURRENT >= ROCSPARSE_VERSION_7_0
#pragma GCC diagnostic pop
#endif
