#include "jacobi_preconditioner.h"
#include <cstdio>

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

  ROCBLAS_CHECK(rocblas_create_handle(&precond_data.handle_rocblas));

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
  
  const double one = 1.0;
  const double minusone = -1.0;
  const double omega = precond_data.jacobi_omega;

  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  // Initialize x = 0
  HIP_CHECK(hipMemset(d_x, 0, n * sizeof(double)));

  for (int iter = 0; iter < precond_data.jacobi_iter; ++iter) {
    // r = b - A*x
    HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));
    
    ROCSPARSE_CHECK(rocsparse_dcsrmv(precond_data.handle_rocsparse,
                                     rocsparse_operation_none,
                                     n,
                                     n,
                                     precond_data.nnz,
                                     &minusone,
                                     A.descr,
                                     A.d_vals,
                                     A.d_row_ptr,
                                     A.d_col_idx,
                                     A.info,
                                     d_x,
                                     &one,
                                     d_r));

    // r = D * r (element-wise)
    elementwise_multiply_kernel<<<num_blocks, block_size>>>(n, precond_data.d_D, d_r);
    HIP_CHECK(hipDeviceSynchronize());

    // x = x + omega * r
    ROCBLAS_CHECK(rocblas_daxpy(precond_data.handle_rocblas, n, &omega, d_r, 1, d_x, 1));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipDeviceSynchronize());

  return 0;
}

void cleanup_jacobi_preconditioner(PreconditionerData& precond_data)
{
  HIP_CHECK(hipFree(precond_data.d_D));
  HIP_CHECK(hipFree(precond_data.d_aux));
  ROCBLAS_CHECK(rocblas_destroy_handle(precond_data.handle_rocblas));
}
