#include "jacobi_preconditioner.h"
#include <cstdio>

// Kernel 1: Compute r = b - A*x using CSR format
__global__ void spmv_residual_kernel(int n,
                                     const int* __restrict__ row_ptr,
                                     const int* __restrict__ col_idx,
                                     const double* __restrict__ vals,
                                     const double* __restrict__ x,
                                     const double* __restrict__ b,
                                     double* __restrict__ r)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < n) {
    double sum = 0.0;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    
    for (int j = row_start; j < row_end; ++j) {
      sum += vals[j] * x[col_idx[j]];
    }
    
    r[row] = b[row] - sum;
  }
}

// Kernel 2: Compute x = x + omega * D^{-1} * r
__global__ void jacobi_update_kernel(int n,
                                     double omega,
                                     const double* __restrict__ D_inv,
                                     const double* __restrict__ r,
                                     double* __restrict__ x)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    x[i] += omega * D_inv[i] * r[i];
  }
}

__global__ void extract_inverted_diagonal_kernel(int n,
                                                  const int* row_ptr,
                                                  const int* col_idx,
                                                  const double* vals,
                                                  double* D)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < n) {
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    
    double diag_val = 0.0;
    for (int j = row_start; j < row_end; ++j) {
      if (col_idx[j] == row) {
        diag_val = vals[j];
        break;
      }
    }
    
    if (diag_val != 0.0) {
      D[row] = 1.0 / diag_val;
    } else {
      D[row] = 1.0;
    }
  }
}

__global__ void elementwise_multiply_kernel(int n,
                                            const double* D,
                                            double* r)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < n) {
    r[i] = D[i] * r[i];
  }
}


int setup_jacobi_preconditioner(const CSRMatrix& A,
                                PreconditionerData& precond_data,
                                rocsparse_handle handle_rocsparse)
{
  precond_data.name = "jacobi";
  precond_data.handle_rocsparse = handle_rocsparse;
  precond_data.n = A.n;
  precond_data.nnz = A.nnz;
  precond_data.d_row_ptr = A.d_row_ptr;
  precond_data.d_col_idx = A.d_col_idx;
  precond_data.A_ptr = &A;
  if (precond_data.jacobi_iter < 1) {
    precond_data.jacobi_iter = 3;  // default value
  }

  HIP_CHECK(hipMalloc(&precond_data.d_D, sizeof(double) * A.n));
  HIP_CHECK(hipMalloc(&precond_data.d_aux, sizeof(double) * A.n));

  // Allocate device scalars
  double h_one = 1.0, h_minusone = -1.0;
  HIP_CHECK(hipMalloc(&precond_data.d_one, sizeof(double)));
  HIP_CHECK(hipMalloc(&precond_data.d_minusone, sizeof(double)));
  HIP_CHECK(hipMalloc(&precond_data.d_zero, sizeof(double)));  // for omega
  HIP_CHECK(hipMemcpy(precond_data.d_one, &h_one, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(precond_data.d_minusone, &h_minusone, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(precond_data.d_zero, &precond_data.jacobi_omega, sizeof(double), hipMemcpyHostToDevice));

  ROCBLAS_CHECK(rocblas_create_handle(&precond_data.handle_rocblas));
  ROCBLAS_CHECK(rocblas_set_pointer_mode(precond_data.handle_rocblas, rocblas_pointer_mode_device));

  int block_size = 256;
  int num_blocks = (A.n + block_size - 1) / block_size;

  extract_inverted_diagonal_kernel<<<num_blocks, block_size>>>(
    A.n,
    A.d_row_ptr,
    A.d_col_idx,
    A.d_vals,
    precond_data.d_D
  );

  HIP_CHECK(hipDeviceSynchronize());

  return 0;
}

int apply_jacobi_preconditioner(double* d_x,
                                const double* d_b,
                                const CSRMatrix& A,
                                PreconditionerData& precond_data)
{
  int n = precond_data.n;
  double* d_r = precond_data.d_aux;
  double omega = precond_data.jacobi_omega;

  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  // Create dense vector descriptors for spmv
  rocsparse_dnvec_descr vec_x, vec_r;
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x,
                                               n,
                                               d_x,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_r,
                                               n,
                                               d_r,
                                               rocsparse_datatype_f64_r));

  // Initialize x = 0
  HIP_CHECK(hipMemset(d_x, 0, n * sizeof(double)));
  rocsparse_error spmv_error;

  for (int iter = 0; iter < precond_data.jacobi_iter; ++iter) {
    // r = b (copy)
    HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));

    // r = -A*x + r = b - A*x
    ROCSPARSE_CHECK(rocsparse_v2_spmv(precond_data.handle_rocsparse,
                                      A.spmv_descr,
                                      precond_data.d_minusone,
                                      A.spmat,
                                      vec_x,
                                      precond_data.d_one,
                                      vec_r,
                                      rocsparse_v2_spmv_stage_compute,
                                      A.spmv_buffer_size,
                                      A.spmv_buffer,
                                      &spmv_error));
    HIP_CHECK(hipDeviceSynchronize());

    // Kernel 2: x = x + omega * D^{-1} * r
    jacobi_update_kernel<<<num_blocks, block_size>>>(
      n,
      omega,
      precond_data.d_D,
      d_r,
      d_x
    );
    HIP_CHECK(hipDeviceSynchronize());
  }

  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_r));

  return 0;
}

void cleanup_jacobi_preconditioner(PreconditionerData& precond_data)
{
  HIP_CHECK(hipFree(precond_data.d_D));
  HIP_CHECK(hipFree(precond_data.d_aux));
  HIP_CHECK(hipFree(precond_data.d_one));
  HIP_CHECK(hipFree(precond_data.d_minusone));
  HIP_CHECK(hipFree(precond_data.d_zero));
  ROCBLAS_CHECK(rocblas_destroy_handle(precond_data.handle_rocblas));
}
