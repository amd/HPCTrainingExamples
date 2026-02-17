/**
 * Preconditioned Conjugate Gradient with Incomplete Cholesky Preconditioner
 * ROCm Implementation - Algorithm
 * 
 * Compatible with ROCm 6.4+ and ROCm 7.x
 * 
 * Based on cg.m reference implementation
 * Uses rocsparse generic APIs (rocsparse_spsv, rocsparse_spmv)
 */

#include "pcg_ic.h"
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse-version.h>
#include <iostream>
#include <cmath>
#include <cstdio>

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
    double* final_res)
{
    // Allocate device memory for CG vectors
    double *d_r, *d_w, *d_p, *d_q;
    HIP_CHECK(hipMalloc(&d_r, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_w, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_p, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_q, n * sizeof(double)));

    // Allocate memory for IC preconditioner values
    double* d_ic_val;
    HIP_CHECK(hipMalloc(&d_ic_val, nnz * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_ic_val,
                        d_csr_val,
                        nnz * sizeof(double),
                        hipMemcpyDeviceToDevice));

    // Attribute values (need addressable variables)
    rocsparse_fill_mode fill_mode_lower = rocsparse_fill_mode_lower;
    rocsparse_diag_type diag_type_non_unit = rocsparse_diag_type_non_unit;

    // Create sparse matrix descriptor for A (using generic API)
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

    // Create sparse matrix descriptor for L (IC factor) - for L solve
    rocsparse_spmat_descr mat_L;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&mat_L,
                                               n,
                                               n,
                                               nnz,
                                               d_csr_row_ptr,
                                               d_csr_col_ind,
                                               d_ic_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(mat_L,
                                                  rocsparse_spmat_fill_mode,
                                                  &fill_mode_lower,
                                                  sizeof(rocsparse_fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(mat_L,
                                                  rocsparse_spmat_diag_type,
                                                  &diag_type_non_unit,
                                                  sizeof(rocsparse_diag_type)));

    // Create separate sparse matrix descriptor for L^T solve
    // (same data, but separate descriptor for preprocessing)
    rocsparse_spmat_descr mat_Lt;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&mat_Lt,
                                               n,
                                               n,
                                               nnz,
                                               d_csr_row_ptr,
                                               d_csr_col_ind,
                                               d_ic_val,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(mat_Lt,
                                                  rocsparse_spmat_fill_mode,
                                                  &fill_mode_lower,
                                                  sizeof(rocsparse_fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(mat_Lt,
                                                  rocsparse_spmat_diag_type,
                                                  &diag_type_non_unit,
                                                  sizeof(rocsparse_diag_type)));

    // Create dense vector descriptors
    rocsparse_dnvec_descr vec_x, vec_r, vec_w, vec_p, vec_q;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x,
                                                 n,
                                                 d_x,
                                                 rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_r,
                                                 n,
                                                 d_r,
                                                 rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_w,
                                                 n,
                                                 d_w,
                                                 rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_p,
                                                 n,
                                                 d_p,
                                                 rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_q,
                                                 n,
                                                 d_q,
                                                 rocsparse_datatype_f64_r));

    // Temporary vector for preconditioning
    double* d_temp;
    HIP_CHECK(hipMalloc(&d_temp, n * sizeof(double)));
    rocsparse_dnvec_descr vec_temp;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_temp,
                                                 n,
                                                 d_temp,
                                                 rocsparse_datatype_f64_r));

    // Constants
    const double one = 1.0;
    const double neg_one = -1.0;
    const double zero = 0.0;

    // =========================================================================
    // Setup Incomplete Cholesky IC(0) factorization
    // =========================================================================
    
    // Create matrix descriptor for IC0
    rocsparse_mat_descr descr_L;
    ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descr_L));
    ROCSPARSE_CHECK(rocsparse_set_mat_type(descr_L,
                                           rocsparse_matrix_type_general));
    ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descr_L,
                                                 rocsparse_index_base_zero));
    ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_L,
                                                rocsparse_fill_mode_lower));
    ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_L,
                                                rocsparse_diag_type_non_unit));

    rocsparse_mat_info info_ic;
    ROCSPARSE_CHECK(rocsparse_create_mat_info(&info_ic));

    // Get buffer size for IC0
    size_t buffer_size_ic;
    ROCSPARSE_CHECK(rocsparse_dcsric0_buffer_size(handle,
                                                  (rocsparse_int)n,
                                                  (rocsparse_int)nnz,
                                                  descr_L,
                                                  d_ic_val,
                                                  d_csr_row_ptr,
                                                  d_csr_col_ind,
                                                  info_ic,
                                                  &buffer_size_ic));

    // =========================================================================
    // Setup SpSV for triangular solves
    // Separate descriptors for L and L^T (required for different preprocessing)
    // =========================================================================

    // Get buffer size for L solve (no transpose)
    size_t buffer_size_L;
    ROCSPARSE_CHECK(rocsparse_spsv(handle,
                                   rocsparse_operation_none,
                                   &one,
                                   mat_L,
                                   vec_r,
                                   vec_temp,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spsv_alg_default,
                                   rocsparse_spsv_stage_buffer_size,
                                   &buffer_size_L,
                                   nullptr));

    // Get buffer size for L^T solve (transpose, using mat_Lt)
    size_t buffer_size_Lt;
    ROCSPARSE_CHECK(rocsparse_spsv(handle,
                                   rocsparse_operation_transpose,
                                   &one,
                                   mat_Lt,
                                   vec_temp,
                                   vec_w,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spsv_alg_default,
                                   rocsparse_spsv_stage_buffer_size,
                                   &buffer_size_Lt,
                                   nullptr));

    // Allocate single buffer with maximum required size
    size_t buffer_size_max = buffer_size_ic;
    if (buffer_size_L > buffer_size_max) buffer_size_max = buffer_size_L;
    if (buffer_size_Lt > buffer_size_max) buffer_size_max = buffer_size_Lt;

    void* d_buffer;
    HIP_CHECK(hipMalloc(&d_buffer, buffer_size_max));

    // Analyze IC0
    ROCSPARSE_CHECK(rocsparse_dcsric0_analysis(handle,
                                               (rocsparse_int)n,
                                               (rocsparse_int)nnz,
                                               descr_L,
                                               d_ic_val,
                                               d_csr_row_ptr,
                                               d_csr_col_ind,
                                               info_ic,
                                               rocsparse_analysis_policy_reuse,
                                               rocsparse_solve_policy_auto,
                                               d_buffer));

    // Compute IC0 factorization: A ≈ L * L^T
    ROCSPARSE_CHECK(rocsparse_dcsric0(handle,
                                      (rocsparse_int)n,
                                      (rocsparse_int)nnz,
                                      descr_L,
                                      d_ic_val,
                                      d_csr_row_ptr,
                                      d_csr_col_ind,
                                      info_ic,
                                      rocsparse_solve_policy_auto,
                                      d_buffer));

    // Check for zero pivot
    rocsparse_int pivot;
    rocsparse_status pivot_status = rocsparse_csric0_zero_pivot(handle,
                                                                info_ic,
                                                                &pivot);
    if (pivot_status == rocsparse_status_zero_pivot) {
        std::cerr << "Warning: IC0 factorization has zero pivot at position " << pivot << std::endl;
    }

    // Preprocess L solve (preprocessing data stored in mat_L descriptor)
    ROCSPARSE_CHECK(rocsparse_spsv(handle,
                                   rocsparse_operation_none,
                                   &one,
                                   mat_L,
                                   vec_r,
                                   vec_temp,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spsv_alg_default,
                                   rocsparse_spsv_stage_preprocess,
                                   &buffer_size_L,
                                   d_buffer));

    // Preprocess L^T solve (preprocessing data stored in mat_Lt descriptor)
    ROCSPARSE_CHECK(rocsparse_spsv(handle,
                                   rocsparse_operation_transpose,
                                   &one,
                                   mat_Lt,
                                   vec_temp,
                                   vec_w,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spsv_alg_default,
                                   rocsparse_spsv_stage_preprocess,
                                   &buffer_size_Lt,
                                   d_buffer));

    // =========================================================================
    // Setup SpMV for matrix-vector products
    // =========================================================================
    
    size_t buffer_size_mv;
    ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                   rocsparse_operation_none,
                                   &one,
                                   mat_A,
                                   vec_p,
                                   &zero,
                                   vec_q,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spmv_alg_default,
                                   rocsparse_spmv_stage_buffer_size,
                                   &buffer_size_mv,
                                   nullptr));

    void* d_buffer_mv;
    HIP_CHECK(hipMalloc(&d_buffer_mv, buffer_size_mv));

    // =========================================================================
    // CG Iteration
    // =========================================================================

    // Compute initial residual: r = b - A * x0
    HIP_CHECK(hipMemcpy(d_r,
                        d_b,
                        n * sizeof(double),
                        hipMemcpyDeviceToDevice));

    // Create temporary vector descriptor for b
    rocsparse_dnvec_descr vec_b;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_b,
                                                 n,
                                                 (void*)d_b,
                                                 rocsparse_datatype_f64_r));

    // r = -A*x + r
    ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                   rocsparse_operation_none,
                                   &neg_one,
                                   mat_A,
                                   vec_x,
                                   &one,
                                   vec_r,
                                   rocsparse_datatype_f64_r,
                                   rocsparse_spmv_alg_default,
                                   rocsparse_spmv_stage_compute,
                                   &buffer_size_mv,
                                   d_buffer_mv));

    // Compute initial residual norm
    double res_init, res_curr;
    ROCBLAS_CHECK(rocblas_ddot(blas_handle,
                               n,
                               d_r,
                               1,
                               d_r,
                               1,
                               &res_init));
    res_init = std::sqrt(res_init);
    
    printf("it %d norm %5.5e\n", 0, res_init);

    if (res_init < 1e-16) {
        *iter = 0;
        *final_res = res_init;
        goto cleanup;
    }

    double rho_current, rho_previous;
    double alpha, beta;
    double pq_dot;

    for (int k = 1; k <= max_iter; k++) {
        // Apply preconditioner: w = M^{-1} * r = (L * L^T)^{-1} * r
        // Step 1: Solve L * temp = r
        ROCSPARSE_CHECK(rocsparse_spsv(handle,
                                       rocsparse_operation_none,
                                       &one,
                                       mat_L,
                                       vec_r,
                                       vec_temp,
                                       rocsparse_datatype_f64_r,
                                       rocsparse_spsv_alg_default,
                                       rocsparse_spsv_stage_compute,
                                       &buffer_size_L,
                                       d_buffer));

        // Step 2: Solve L^T * w = temp
        ROCSPARSE_CHECK(rocsparse_spsv(handle,
                                       rocsparse_operation_transpose,
                                       &one,
                                       mat_Lt,
                                       vec_temp,
                                       vec_w,
                                       rocsparse_datatype_f64_r,
                                       rocsparse_spsv_alg_default,
                                       rocsparse_spsv_stage_compute,
                                       &buffer_size_Lt,
                                       d_buffer));

        // rho_current = r' * w
        ROCBLAS_CHECK(rocblas_ddot(blas_handle,
                                   n,
                                   d_r,
                                   1,
                                   d_w,
                                   1,
                                   &rho_current));

        if (k == 1) {
            // p = w
            HIP_CHECK(hipMemcpy(d_p,
                                d_w,
                                n * sizeof(double),
                                hipMemcpyDeviceToDevice));
        } else {
            // beta = rho_current / rho_previous
            beta = rho_current / rho_previous;
            // p = w + beta * p
            ROCBLAS_CHECK(rocblas_dscal(blas_handle,
                                        n,
                                        &beta,
                                        d_p,
                                        1));
            ROCBLAS_CHECK(rocblas_daxpy(blas_handle,
                                        n,
                                        &one,
                                        d_w,
                                        1,
                                        d_p,
                                        1));
        }

        // q = A * p
        ROCSPARSE_CHECK(rocsparse_spmv(handle,
                                       rocsparse_operation_none,
                                       &one,
                                       mat_A,
                                       vec_p,
                                       &zero,
                                       vec_q,
                                       rocsparse_datatype_f64_r,
                                       rocsparse_spmv_alg_default,
                                       rocsparse_spmv_stage_compute,
                                       &buffer_size_mv,
                                       d_buffer_mv));

        // alpha = rho_current / (p' * q)
        ROCBLAS_CHECK(rocblas_ddot(blas_handle,
                                   n,
                                   d_p,
                                   1,
                                   d_q,
                                   1,
                                   &pq_dot));
        alpha = rho_current / pq_dot;

        // x = x + alpha * p
        ROCBLAS_CHECK(rocblas_daxpy(blas_handle,
                                    n,
                                    &alpha,
                                    d_p,
                                    1,
                                    d_x,
                                    1));

        // r = r - alpha * q
        double neg_alpha = -alpha;
        ROCBLAS_CHECK(rocblas_daxpy(blas_handle,
                                    n,
                                    &neg_alpha,
                                    d_q,
                                    1,
                                    d_r,
                                    1));

        // Compute residual norm
        ROCBLAS_CHECK(rocblas_ddot(blas_handle,
                                   n,
                                   d_r,
                                   1,
                                   d_r,
                                   1,
                                   &res_curr));
        res_curr = std::sqrt(res_curr);

        printf("it %d norm %5.5e\n", k, res_curr);

        // Check convergence
        if (res_curr / res_init < tol) {
            *iter = k;
            *final_res = res_curr;
            goto cleanup;
        }

        rho_previous = rho_current;
    }

    *iter = max_iter;
    *final_res = res_curr;

cleanup:
    // Clean up vectors
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_p));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_ic_val));
    HIP_CHECK(hipFree(d_buffer));
    HIP_CHECK(hipFree(d_buffer_mv));

    // Destroy descriptors
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(mat_A));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(mat_L));
    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(mat_Lt));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_r));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_w));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_p));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_q));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_temp));
    ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_b));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_L));
    ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info_ic));
}

// Restore warning settings
#if ROCSPARSE_VERSION_CURRENT >= ROCSPARSE_VERSION_7_0
#pragma GCC diagnostic pop
#endif
