#include "jacobi_preconditioner.h"
#include <cstdio>

int setup_jacobi_preconditioner(const CSRMatrix& A,
                                PreconditionerData& precond_data,
                                rocsparse_handle handle_rocsparse)
{
  precond_data.name = "it_jacobi";
  precond_data.handle_rocsparse = handle_rocsparse;
  precond_data.n = A.n;
  precond_data.nnz = A.nnz;
  precond_data.d_row_ptr = A.d_row_ptr;
  precond_data.d_col_idx = A.d_col_idx;

  // TODO: implement setup

  return 0;
}

int apply_jacobi_preconditioner(double* d_x,
                                const double* d_r,
                                const CSRMatrix& A,
                                PreconditionerData& precond_data)
{
  // TODO: implement apply

  return 0;
}

void cleanup_jacobi_preconditioner(PreconditionerData& precond_data)
{
  // TODO: implement cleanup
}
