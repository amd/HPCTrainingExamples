#include "asynch_jacobi_preconditioner.h"
#include "jacobi_preconditioner.h"
#include <cstdio>

__global__ void apply_asynch_jacobi_preconditioner_static_kernel(int n,
								 int jacobi_iter,
								 double omega,
								 double* x,
								 const double* b,
								 const int* row_ptr,
								 const int* col_idx,
								 const double* vals,
								 const double* D)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int subwf_id = tid / SUBWF_SIZE;
  int num_subwf = (blockDim.x * gridDim.x) / SUBWF_SIZE;
  int tid_in_subwf = tid % SUBWF_SIZE;
  int row = subwf_id;
  int iter = 0;
  if (row < n) { 
    int p = row_ptr[row];
    int q = row_ptr[row + 1];
    double brow = b[row];
    double Drow = D[row];
    while (iter < jacobi_iter) {
      double sum = 0.0;
      for (int i = p + tid_in_subwf; i < q; i += SUBWF_SIZE) {
	sum += vals[i] * x[col_idx[i]];
      }

#pragma unroll
      for (int i = SUBWF_SIZE >> 1; i > 0; i >>= 1)
	sum += __shfl_down(sum, i, SUBWF_SIZE);

      if (!tid_in_subwf) {
	x[row] += omega * Drow * (brow - sum);
      }
      __threadfence();
      iter++;
    }
  }
}

__global__ void apply_asynch_jacobi_preconditioner_dynamic_kernel(int n,
								  int jacobi_iter,
								  double omega,
								  double* x,
								  const double* b,
								  const int* row_ptr,
								  const int* col_idx,
								  const double* vals,
								  const double* D)
{

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int subwf_id = tid / SUBWF_SIZE;
  int num_subwf = (blockDim.x * gridDim.x) / SUBWF_SIZE;
  int tid_in_subwf = tid % SUBWF_SIZE;
  int row = subwf_id;
  int iter = 0;
  while (iter < jacobi_iter) {
    int p = row_ptr[row];
    int q = row_ptr[row + 1];
    double sum = 0.0;
    for (int i = p + tid_in_subwf; i < q; i += SUBWF_SIZE) {
      sum += vals[i] * x[col_idx[i]];
    }

#pragma unroll
    for (int i = SUBWF_SIZE >> 1; i > 0; i >>= 1)
      sum += __shfl_down(sum, i, SUBWF_SIZE);

    if (!tid_in_subwf) {
      x[row] += omega * D[row] * (b[row] - sum);
    }
//    __threadfence();
    row += num_subwf;
    if (row >=n) {
      row -= n;
      iter++;
    }
  }
}

int setup_asynch_jacobi_preconditioner(const CSRMatrix& A,
				       PreconditionerData& precond_data,
				       rocsparse_handle handle_rocsparse)
{
  int version = precond_data.asynch_jacobi_version;
  if (version < 0 || version > 1) {
    printf("Error: Invalid asynch_jacobi_version: %d\n", version);
    printf("Valid options are: 0, 1\n");
    return -1;
  }

  int status = setup_jacobi_preconditioner(A, precond_data, handle_rocsparse);
  if (status != 0) {
    return status;
  }

  precond_data.name = "asynch_jacobi";

  // TODO: add asynch-specific setup here

  return 0;
}

int apply_asynch_jacobi_preconditioner(double* d_x,
				       const double* d_b,
				       const CSRMatrix& A,
				       PreconditionerData& precond_data)
{
  int n = precond_data.n;
  int version = precond_data.asynch_jacobi_version;


  HIP_CHECK(hipMemset(d_x, 0, n * sizeof(double)));

  if (version == 0) {
    int block_size = 1024;
    int num_blocks = 512;
    apply_asynch_jacobi_preconditioner_dynamic_kernel<<<num_blocks, block_size>>>(
										  n,
										  precond_data.jacobi_iter,
										  precond_data.jacobi_omega,
										  d_x,
										  d_b,
										  A.d_row_ptr,
										  A.d_col_idx,
										  A.d_vals,
										  precond_data.d_D
										 );
  } else if (version == 1) {
    int block_size = 1024;
    int threads_needed = n * SUBWF_SIZE;
    int num_blocks = (threads_needed + block_size - 1) / block_size;
    apply_asynch_jacobi_preconditioner_static_kernel<<<num_blocks, block_size>>>(
										 n,
										 precond_data.jacobi_iter,
										 precond_data.jacobi_omega,
										 d_x,
										 d_b,
										 A.d_row_ptr,
										 A.d_col_idx,
										 A.d_vals,
										 precond_data.d_D
										);
  }

  HIP_CHECK(hipDeviceSynchronize());

  return 0;
}

void cleanup_asynch_jacobi_preconditioner(PreconditionerData& precond_data)
{
  cleanup_jacobi_preconditioner(precond_data);

  // TODO: add asynch-specific cleanup here
}
