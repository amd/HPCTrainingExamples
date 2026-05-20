#include "iterative_gauss_seidel.h"
#include <cstdio>
#include <cstdlib>

__global__ void extract_L_U_D_kernel(int n,
                                     const int* A_row_ptr,
                                     const int* A_col_idx,
                                     const double* A_vals,
                                     int* L_row_ptr,
                                     int* L_col_idx,
                                     double* L_vals,
                                     int* U_row_ptr,
                                     int* U_col_idx,
                                     double* U_vals,
                                     double* D)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < n) {
    int A_start = A_row_ptr[row];
    int A_end = A_row_ptr[row + 1];
    
    int L_start = L_row_ptr[row];
    int U_start = U_row_ptr[row];
    
    int L_idx = L_start;
    int U_idx = U_start;
    
    for (int j = A_start; j < A_end; ++j) {
      int col = A_col_idx[j];
      double val = A_vals[j];
      
      if (col < row) {
        L_col_idx[L_idx] = col;
        L_vals[L_idx] = val;
        L_idx++;
      } else if (col > row) {
        U_col_idx[U_idx] = col;
        U_vals[U_idx] = val;
        U_idx++;
      } else {
        D[row] = 1.0 / val;
      }
    }
  }
}

__global__ void count_L_U_nnz_kernel(int n,
                                     const int* A_row_ptr,
                                     const int* A_col_idx,
                                     int* L_row_counts,
                                     int* U_row_counts)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < n) {
    int A_start = A_row_ptr[row];
    int A_end = A_row_ptr[row + 1];
    
    int L_count = 0;
    int U_count = 0;
    
    for (int j = A_start; j < A_end; ++j) {
      int col = A_col_idx[j];
      if (col < row) {
        L_count++;
      } else if (col > row) {
        U_count++;
      }
    }
    
    L_row_counts[row] = L_count;
    U_row_counts[row] = U_count;
  }
}

__global__ void elementwise_multiply_gs_kernel(int n,
                                               const double* a,
                                               const double* b,
                                               double* c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] * b[i];
  }
}

int setup_gs_preconditioner(const CSRMatrix& A,
                            PreconditionerData& precond_data,
                            rocsparse_handle handle_rocsparse)
{
  int n = A.n;
  precond_data.handle_rocsparse = handle_rocsparse;
  precond_data.n = n;
  precond_data.nnz = A.nnz;
  precond_data.d_row_ptr = A.d_row_ptr;
  precond_data.d_col_idx = A.d_col_idx;
  precond_data.A_ptr = &A;

  if (precond_data.gs_inner_iter < 1) {
    precond_data.gs_inner_iter = 3;
  }
  if (precond_data.gs_outer_iter < 1) {
    precond_data.gs_outer_iter = 1;
  }

  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  // Count L and U nnz per row
  int* d_L_row_counts;
  int* d_U_row_counts;
  HIP_CHECK(hipMalloc(&d_L_row_counts, sizeof(int) * n));
  HIP_CHECK(hipMalloc(&d_U_row_counts, sizeof(int) * n));

  count_L_U_nnz_kernel<<<num_blocks, block_size>>>(
    n, A.d_row_ptr, A.d_col_idx, d_L_row_counts, d_U_row_counts);
  HIP_CHECK(hipDeviceSynchronize());

  // Copy counts to host and compute row pointers
  int* h_L_row_counts = (int*) malloc(sizeof(int) * n);
  int* h_U_row_counts = (int*) malloc(sizeof(int) * n);
  int* h_L_row_ptr = (int*) malloc(sizeof(int) * (n + 1));
  int* h_U_row_ptr = (int*) malloc(sizeof(int) * (n + 1));

  HIP_CHECK(hipMemcpy(h_L_row_counts, d_L_row_counts, sizeof(int) * n, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(h_U_row_counts, d_U_row_counts, sizeof(int) * n, hipMemcpyDeviceToHost));

  h_L_row_ptr[0] = 0;
  h_U_row_ptr[0] = 0;
  for (int i = 0; i < n; ++i) {
    h_L_row_ptr[i + 1] = h_L_row_ptr[i] + h_L_row_counts[i];
    h_U_row_ptr[i + 1] = h_U_row_ptr[i] + h_U_row_counts[i];
  }
  precond_data.L_nnz = h_L_row_ptr[n];
  precond_data.U_nnz = h_U_row_ptr[n];

  // Allocate L and U matrices
  HIP_CHECK(hipMalloc(&precond_data.d_L_row_ptr, sizeof(int) * (n + 1)));
  HIP_CHECK(hipMalloc(&precond_data.d_L_col_idx, sizeof(int) * (precond_data.L_nnz > 0 ? precond_data.L_nnz : 1)));
  HIP_CHECK(hipMalloc(&precond_data.d_L_vals, sizeof(double) * (precond_data.L_nnz > 0 ? precond_data.L_nnz : 1)));

  HIP_CHECK(hipMalloc(&precond_data.d_U_row_ptr, sizeof(int) * (n + 1)));
  HIP_CHECK(hipMalloc(&precond_data.d_U_col_idx, sizeof(int) * (precond_data.U_nnz > 0 ? precond_data.U_nnz : 1)));
  HIP_CHECK(hipMalloc(&precond_data.d_U_vals, sizeof(double) * (precond_data.U_nnz > 0 ? precond_data.U_nnz : 1)));

  HIP_CHECK(hipMemcpy(precond_data.d_L_row_ptr, h_L_row_ptr, sizeof(int) * (n + 1), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(precond_data.d_U_row_ptr, h_U_row_ptr, sizeof(int) * (n + 1), hipMemcpyHostToDevice));

  // Allocate D (inverted diagonal)
  HIP_CHECK(hipMalloc(&precond_data.d_D, sizeof(double) * n));

  // Extract L, U, D
  extract_L_U_D_kernel<<<num_blocks, block_size>>>(
    n,
    A.d_row_ptr, A.d_col_idx, A.d_vals,
    precond_data.d_L_row_ptr, precond_data.d_L_col_idx, precond_data.d_L_vals,
    precond_data.d_U_row_ptr, precond_data.d_U_col_idx, precond_data.d_U_vals,
    precond_data.d_D);
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate auxiliary vectors
  HIP_CHECK(hipMalloc(&precond_data.d_aux_vec1, sizeof(double) * n));
  HIP_CHECK(hipMalloc(&precond_data.d_aux_vec2, sizeof(double) * n));
  HIP_CHECK(hipMalloc(&precond_data.d_aux_vec3, sizeof(double) * n));

  // Create modern rocsparse_v2_spmv descriptors for L
  ROCSPARSE_CHECK(rocsparse_create_csr_descr(&precond_data.spmatL,
                                             n,
                                             n,
                                             precond_data.L_nnz,
                                             precond_data.d_L_row_ptr,
                                             precond_data.d_L_col_idx,
                                             precond_data.d_L_vals,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_r));

  // Create modern rocsparse_v2_spmv descriptors for U
  ROCSPARSE_CHECK(rocsparse_create_csr_descr(&precond_data.spmatU,
                                             n,
                                             n,
                                             precond_data.U_nnz,
                                             precond_data.d_U_row_ptr,
                                             precond_data.d_U_col_idx,
                                             precond_data.d_U_vals,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_r));

  // Create spmv descriptors for L and U
  ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&precond_data.spmv_descr_L));
  ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&precond_data.spmv_descr_U));

  rocsparse_operation op = rocsparse_operation_none;
  rocsparse_spmv_alg alg = rocsparse_spmv_alg_default;
  rocsparse_datatype compute_type = rocsparse_datatype_f64_r;
  rocsparse_error set_input_error;
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_L,
                                           rocsparse_spmv_input_operation,
                                           &op,
                                           sizeof(op),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_L,
                                           rocsparse_spmv_input_alg,
                                           &alg,
                                           sizeof(alg),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_L,
                                           rocsparse_spmv_input_scalar_datatype,
                                           &compute_type,
                                           sizeof(compute_type),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_L,
                                           rocsparse_spmv_input_compute_datatype,
                                           &compute_type,
                                           sizeof(compute_type),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_U,
                                           rocsparse_spmv_input_operation,
                                           &op,
                                           sizeof(op),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_U,
                                           rocsparse_spmv_input_alg,
                                           &alg,
                                           sizeof(alg),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_U,
                                           rocsparse_spmv_input_scalar_datatype,
                                           &compute_type,
                                           sizeof(compute_type),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           precond_data.spmv_descr_U,
                                           rocsparse_spmv_input_compute_datatype,
                                           &compute_type,
                                           sizeof(compute_type),
                                           &set_input_error));

  // Set up spmv buffers for L and U
  rocsparse_dnvec_descr tmp_x, tmp_y;
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&tmp_x,
                                               n,
                                               precond_data.d_aux_vec1,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&tmp_y,
                                               n,
                                               precond_data.d_aux_vec2,
                                               rocsparse_datatype_f64_r));

  double h_alpha = 1.0, h_beta = 0.0;
  rocsparse_error spmv_error;

  // Buffer size for L
  ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle_rocsparse,
                                                precond_data.spmv_descr_L,
                                                precond_data.spmatL,
                                                tmp_x,
                                                tmp_y,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &precond_data.spmv_buffer_size_L,
                                                &spmv_error));
  HIP_CHECK(hipMalloc(&precond_data.spmv_buffer_L, precond_data.spmv_buffer_size_L));

  // Analysis for L
  ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                    precond_data.spmv_descr_L,
                                    &h_alpha,
                                    precond_data.spmatL,
                                    tmp_x,
                                    &h_beta,
                                    tmp_y,
                                    rocsparse_v2_spmv_stage_analysis,
                                    precond_data.spmv_buffer_size_L,
                                    precond_data.spmv_buffer_L,
                                    &spmv_error));

  // Buffer size for U
  ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle_rocsparse,
                                                precond_data.spmv_descr_U,
                                                precond_data.spmatU,
                                                tmp_x,
                                                tmp_y,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &precond_data.spmv_buffer_size_U,
                                                &spmv_error));
  HIP_CHECK(hipMalloc(&precond_data.spmv_buffer_U, precond_data.spmv_buffer_size_U));

  // Analysis for U
  ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                    precond_data.spmv_descr_U,
                                    &h_alpha,
                                    precond_data.spmatU,
                                    tmp_x,
                                    &h_beta,
                                    tmp_y,
                                    rocsparse_v2_spmv_stage_analysis,
                                    precond_data.spmv_buffer_size_U,
                                    precond_data.spmv_buffer_U,
                                    &spmv_error));

  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(tmp_x));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(tmp_y));

  // Create rocblas handle
  ROCBLAS_CHECK(rocblas_create_handle(&precond_data.handle_rocblas));

  // Allocate device scalars for fully async operations
  double h_one = 1.0, h_zero = 0.0, h_minusone = -1.0;
  HIP_CHECK(hipMalloc(&precond_data.d_one, sizeof(double)));
  HIP_CHECK(hipMalloc(&precond_data.d_zero, sizeof(double)));
  HIP_CHECK(hipMalloc(&precond_data.d_minusone, sizeof(double)));
  HIP_CHECK(hipMemcpy(precond_data.d_one, &h_one, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(precond_data.d_zero, &h_zero, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(precond_data.d_minusone, &h_minusone, sizeof(double), hipMemcpyHostToDevice));

  // Set rocblas to use device pointers for scalars
  ROCBLAS_CHECK(rocblas_set_pointer_mode(precond_data.handle_rocblas, rocblas_pointer_mode_device));

  // Cleanup temporary
  HIP_CHECK(hipFree(d_L_row_counts));
  HIP_CHECK(hipFree(d_U_row_counts));
  free(h_L_row_counts);
  free(h_U_row_counts);
  free(h_L_row_ptr);
  free(h_U_row_ptr);

  return 0;
}

int apply_gs_it_preconditioner(double* d_x,
                               const double* d_b,
                               const CSRMatrix& A,
                               PreconditionerData& precond_data)
{
  int n = precond_data.n;
  int k = precond_data.gs_inner_iter;
  int m = precond_data.gs_outer_iter;

  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  // Create dense vector descriptors for spmv
  rocsparse_dnvec_descr vec_x, vec_aux1, vec_aux2, vec_aux3;
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x,
                                               n,
                                               d_x,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_aux1,
                                               n,
                                               precond_data.d_aux_vec1,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_aux2,
                                               n,
                                               precond_data.d_aux_vec2,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_aux3,
                                               n,
                                               precond_data.d_aux_vec3,
                                               rocsparse_datatype_f64_r));

  // x = 0
  HIP_CHECK(hipMemset(d_x, 0, n * sizeof(double)));
  rocsparse_error spmv_error;

  // Outer loop
  for (int j = 0; j < m; ++j) {
    // r = b - A*x -> aux_vec2
    HIP_CHECK(hipMemcpy(precond_data.d_aux_vec2, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));
    ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                      A.spmv_descr,
                                      precond_data.d_minusone,
                                      A.spmat,
                                      vec_x,
                                      precond_data.d_one,
                                      vec_aux2,
                                      rocsparse_v2_spmv_stage_compute,
                                      A.spmv_buffer_size,
                                      A.spmv_buffer,
                                      &spmv_error));

    // y = D^{-1} * r -> aux_vec1
    elementwise_multiply_gs_kernel<<<num_blocks, block_size>>>(
      n, precond_data.d_aux_vec2, precond_data.d_D, precond_data.d_aux_vec1);

    // Inner loop 1: forward sweep with L
    for (int i = 0; i < k; ++i) {
      // y = D^{-1} * (r - L*y)
      HIP_CHECK(hipMemcpy(precond_data.d_aux_vec3, precond_data.d_aux_vec2, sizeof(double) * n, hipMemcpyDeviceToDevice));
      ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                        precond_data.spmv_descr_L,
                                        precond_data.d_minusone,
                                        precond_data.spmatL,
                                        vec_aux1,
                                        precond_data.d_one,
                                        vec_aux3,
                                        rocsparse_v2_spmv_stage_compute,
                                        precond_data.spmv_buffer_size_L,
                                        precond_data.spmv_buffer_L,
                                        &spmv_error));
      elementwise_multiply_gs_kernel<<<num_blocks, block_size>>>(
        n, precond_data.d_aux_vec3, precond_data.d_D, precond_data.d_aux_vec1);
    }

    // Inner loop 2: backward sweep with U
    for (int i = 0; i < k; ++i) {
      // y = D^{-1} * (r - U*y)
      HIP_CHECK(hipMemcpy(precond_data.d_aux_vec3, precond_data.d_aux_vec2, sizeof(double) * n, hipMemcpyDeviceToDevice));
      ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                        precond_data.spmv_descr_U,
                                        precond_data.d_minusone,
                                        precond_data.spmatU,
                                        vec_aux1,
                                        precond_data.d_one,
                                        vec_aux3,
                                        rocsparse_v2_spmv_stage_compute,
                                        precond_data.spmv_buffer_size_U,
                                        precond_data.spmv_buffer_U,
                                        &spmv_error));
      elementwise_multiply_gs_kernel<<<num_blocks, block_size>>>(
        n, precond_data.d_aux_vec3, precond_data.d_D, precond_data.d_aux_vec1);
    }

    // x = x + y
    ROCBLAS_CHECK(rocblas_daxpy(precond_data.handle_rocblas,
                                n,
                                precond_data.d_one,
                                precond_data.d_aux_vec1,
                                1,
                                d_x,
                                1));
  }

  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_aux1));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_aux2));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_aux3));

  HIP_CHECK(hipDeviceSynchronize());
  return 0;
}

int apply_gs_it2_preconditioner(double* d_x,
                                const double* d_b,
                                const CSRMatrix& A,
                                PreconditionerData& precond_data)
{
  int n = precond_data.n;
  int k = precond_data.gs_inner_iter;
  int m = precond_data.gs_outer_iter;

  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  // Create dense vector descriptors for spmv
  rocsparse_dnvec_descr vec_aux1, vec_aux2, vec_aux3;
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_aux1,
                                               n,
                                               precond_data.d_aux_vec1,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_aux2,
                                               n,
                                               precond_data.d_aux_vec2,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_aux3,
                                               n,
                                               precond_data.d_aux_vec3,
                                               rocsparse_datatype_f64_r));

  // y = D^{-1} * b -> aux_vec1
  elementwise_multiply_gs_kernel<<<num_blocks, block_size>>>(
    n, d_b, precond_data.d_D, precond_data.d_aux_vec1);

  rocsparse_error spmv_error;

  // Outer loop
  for (int j = 0; j < m; ++j) {
    // Inner loop 1
    for (int i = 0; i < 1; ++i) {
      // aux_vec2 = L * y
      ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                        precond_data.spmv_descr_L,
                                        precond_data.d_one,
                                        precond_data.spmatL,
                                        vec_aux1,
                                        precond_data.d_zero,
                                        vec_aux2,
                                        rocsparse_v2_spmv_stage_compute,
                                        precond_data.spmv_buffer_size_L,
                                        precond_data.spmv_buffer_L,
                                        &spmv_error));

      // aux_vec3 = U * y
      ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                        precond_data.spmv_descr_U,
                                        precond_data.d_one,
                                        precond_data.spmatU,
                                        vec_aux1,
                                        precond_data.d_zero,
                                        vec_aux3,
                                        rocsparse_v2_spmv_stage_compute,
                                        precond_data.spmv_buffer_size_U,
                                        precond_data.spmv_buffer_U,
                                        &spmv_error));

      // aux_vec2 = aux_vec2 + aux_vec3 = (L+U)*y
      ROCBLAS_CHECK(rocblas_daxpy(precond_data.handle_rocblas,
                                  n,
                                  precond_data.d_one,
                                  precond_data.d_aux_vec3,
                                  1,
                                  precond_data.d_aux_vec2,
                                  1));

      // aux_vec3 = b
      HIP_CHECK(hipMemcpy(precond_data.d_aux_vec3, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));

      // aux_vec3 = b - (L+U)*y
      ROCBLAS_CHECK(rocblas_daxpy(precond_data.handle_rocblas,
                                  n,
                                  precond_data.d_minusone,
                                  precond_data.d_aux_vec2,
                                  1,
                                  precond_data.d_aux_vec3,
                                  1));

      // aux_vec2 = D^{-1} * aux_vec3
      elementwise_multiply_gs_kernel<<<num_blocks, block_size>>>(
        n, precond_data.d_aux_vec3, precond_data.d_D, precond_data.d_aux_vec2);
    }

    // r = b - L*y -> aux_vec3
    HIP_CHECK(hipMemcpy(precond_data.d_aux_vec3, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));
    ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                      precond_data.spmv_descr_L,
                                      precond_data.d_minusone,
                                      precond_data.spmatL,
                                      vec_aux2,
                                      precond_data.d_one,
                                      vec_aux3,
                                      rocsparse_v2_spmv_stage_compute,
                                      precond_data.spmv_buffer_size_L,
                                      precond_data.spmv_buffer_L,
                                      &spmv_error));

    // Inner loop 2: y = D^{-1} * (r - U*y)
    for (int i = 0; i < k; ++i) {
      HIP_CHECK(hipMemcpy(precond_data.d_aux_vec1, precond_data.d_aux_vec3, sizeof(double) * n, hipMemcpyDeviceToDevice));
      ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                        precond_data.spmv_descr_U,
                                        precond_data.d_minusone,
                                        precond_data.spmatU,
                                        vec_aux2,
                                        precond_data.d_one,
                                        vec_aux1,
                                        rocsparse_v2_spmv_stage_compute,
                                        precond_data.spmv_buffer_size_U,
                                        precond_data.spmv_buffer_U,
                                        &spmv_error));
      elementwise_multiply_gs_kernel<<<num_blocks, block_size>>>(
        n, precond_data.d_D, precond_data.d_aux_vec1, precond_data.d_aux_vec2);
    }

    // r = b - U*y -> aux_vec3
    HIP_CHECK(hipMemcpy(precond_data.d_aux_vec3, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));
    ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                      precond_data.spmv_descr_U,
                                      precond_data.d_minusone,
                                      precond_data.spmatU,
                                      vec_aux2,
                                      precond_data.d_one,
                                      vec_aux3,
                                      rocsparse_v2_spmv_stage_compute,
                                      precond_data.spmv_buffer_size_U,
                                      precond_data.spmv_buffer_U,
                                      &spmv_error));

    // Inner loop 3: y = D^{-1} * (r - L*y)
    for (int i = 0; i < k; ++i) {
      HIP_CHECK(hipMemcpy(precond_data.d_aux_vec1, precond_data.d_aux_vec3, sizeof(double) * n, hipMemcpyDeviceToDevice));
      ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                        precond_data.spmv_descr_L,
                                        precond_data.d_minusone,
                                        precond_data.spmatL,
                                        vec_aux2,
                                        precond_data.d_one,
                                        vec_aux1,
                                        rocsparse_v2_spmv_stage_compute,
                                        precond_data.spmv_buffer_size_L,
                                        precond_data.spmv_buffer_L,
                                        &spmv_error));
      elementwise_multiply_gs_kernel<<<num_blocks, block_size>>>(
        n, precond_data.d_D, precond_data.d_aux_vec1, precond_data.d_aux_vec2);
    }

    HIP_CHECK(hipMemcpy(precond_data.d_aux_vec1, precond_data.d_aux_vec2, sizeof(double) * n, hipMemcpyDeviceToDevice));
  }

  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_aux1));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_aux2));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_aux3));

  HIP_CHECK(hipMemcpy(d_x, precond_data.d_aux_vec2, sizeof(double) * n, hipMemcpyDeviceToDevice));
//  HIP_CHECK(hipDeviceSynchronize());
  return 0;
}

void cleanup_gs_preconditioner(PreconditionerData& precond_data)
{
  HIP_CHECK(hipFree(precond_data.d_L_vals));
  HIP_CHECK(hipFree(precond_data.d_L_row_ptr));
  HIP_CHECK(hipFree(precond_data.d_L_col_idx));

  HIP_CHECK(hipFree(precond_data.d_U_vals));
  HIP_CHECK(hipFree(precond_data.d_U_row_ptr));
  HIP_CHECK(hipFree(precond_data.d_U_col_idx));

  HIP_CHECK(hipFree(precond_data.d_D));

  HIP_CHECK(hipFree(precond_data.d_aux_vec1));
  HIP_CHECK(hipFree(precond_data.d_aux_vec2));
  HIP_CHECK(hipFree(precond_data.d_aux_vec3));

  HIP_CHECK(hipFree(precond_data.d_one));
  HIP_CHECK(hipFree(precond_data.d_zero));
  HIP_CHECK(hipFree(precond_data.d_minusone));

  ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(precond_data.spmatL));
  ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(precond_data.spmatU));
  ROCSPARSE_CHECK(rocsparse_destroy_spmv_descr(precond_data.spmv_descr_L));
  ROCSPARSE_CHECK(rocsparse_destroy_spmv_descr(precond_data.spmv_descr_U));
  HIP_CHECK(hipFree(precond_data.spmv_buffer_L));
  HIP_CHECK(hipFree(precond_data.spmv_buffer_U));

  ROCBLAS_CHECK(rocblas_destroy_handle(precond_data.handle_rocblas));
}
