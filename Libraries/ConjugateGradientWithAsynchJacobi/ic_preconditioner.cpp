#include "ic_preconditioner.h"
#include <cstdio>
#include <algorithm>

int setup_ic_preconditioner(const CSRMatrix& A,
                            PreconditionerData& precond_data,
                            rocsparse_handle handle_rocsparse)
{
  precond_data.name = "ic";
  precond_data.handle_rocsparse = handle_rocsparse;
  precond_data.n = A.n;
  precond_data.nnz = A.nnz;
  precond_data.d_row_ptr = A.d_row_ptr;
  precond_data.d_col_idx = A.d_col_idx;

  HIP_CHECK(hipMalloc(&precond_data.d_M_vals, sizeof(double) * A.nnz));
  HIP_CHECK(hipMemcpy(precond_data.d_M_vals, A.d_vals, sizeof(double) * A.nnz, hipMemcpyDeviceToDevice));

  HIP_CHECK(hipMalloc(&precond_data.d_aux, sizeof(double) * A.n));

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&precond_data.descrM));
  ROCSPARSE_CHECK(rocsparse_set_mat_type(precond_data.descrM, rocsparse_matrix_type_general));

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&precond_data.descrL));
  ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(precond_data.descrL, rocsparse_fill_mode_lower));
  ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(precond_data.descrL, rocsparse_diag_type_non_unit));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(precond_data.descrL, rocsparse_index_base_zero));

  ROCSPARSE_CHECK(rocsparse_create_mat_info(&precond_data.infoM));

  size_t buffer_size_M, buffer_size_L, buffer_size_Lt;

  ROCSPARSE_CHECK(rocsparse_dcsric0_buffer_size(handle_rocsparse,
                                                A.n,
                                                A.nnz,
                                                precond_data.descrM,
                                                precond_data.d_M_vals,
                                                A.d_row_ptr,
                                                A.d_col_idx,
                                                precond_data.infoM,
                                                &buffer_size_M));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                                               rocsparse_operation_none,
                                               A.n,
                                               A.nnz,
                                               precond_data.descrL,
                                               precond_data.d_M_vals,
                                               A.d_row_ptr,
                                               A.d_col_idx,
                                               precond_data.infoM,
                                               &buffer_size_L));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
                                               rocsparse_operation_transpose,
                                               A.n,
                                               A.nnz,
                                               precond_data.descrL,
                                               precond_data.d_M_vals,
                                               A.d_row_ptr,
                                               A.d_col_idx,
                                               precond_data.infoM,
                                               &buffer_size_Lt));

  precond_data.buffer_size = std::max(buffer_size_M, std::max(buffer_size_L, buffer_size_Lt));
  HIP_CHECK(hipMalloc(&precond_data.buffer, precond_data.buffer_size));

  ROCSPARSE_CHECK(rocsparse_dcsric0_analysis(handle_rocsparse,
                                             A.n,
                                             A.nnz,
                                             precond_data.descrM,
                                             precond_data.d_M_vals,
                                             A.d_row_ptr,
                                             A.d_col_idx,
                                             precond_data.infoM,
                                             rocsparse_analysis_policy_reuse,
                                             rocsparse_solve_policy_auto,
                                             precond_data.buffer));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
                                            rocsparse_operation_none,
                                            A.n,
                                            A.nnz,
                                            precond_data.descrL,
                                            precond_data.d_M_vals,
                                            A.d_row_ptr,
                                            A.d_col_idx,
                                            precond_data.infoM,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            precond_data.buffer));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
                                            rocsparse_operation_transpose,
                                            A.n,
                                            A.nnz,
                                            precond_data.descrL,
                                            precond_data.d_M_vals,
                                            A.d_row_ptr,
                                            A.d_col_idx,
                                            precond_data.infoM,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            precond_data.buffer));

  rocsparse_int position;
  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
                                                                 precond_data.infoM,
                                                                 &position)) {
    printf("A has structural zero at A(%d,%d). IC setup failed.\n", position, position);
    return -1;
  }

  ROCSPARSE_CHECK(rocsparse_dcsric0(handle_rocsparse,
                                    A.n,
                                    A.nnz,
                                    precond_data.descrM,
                                    precond_data.d_M_vals,
                                    A.d_row_ptr,
                                    A.d_col_idx,
                                    precond_data.infoM,
                                    rocsparse_solve_policy_auto,
                                    precond_data.buffer));

  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
                                                                 precond_data.infoM,
                                                                 &position)) {
    printf("L has structural and/or numerical zero at L(%d,%d). IC setup failed.\n",
           position, position);
    return -1;
  }

  HIP_CHECK(hipDeviceSynchronize());
  return 0;
}

int apply_ic_preconditioner(double* d_x,
                            const double* d_r,
                            const CSRMatrix& A,
                            PreconditionerData& precond_data)
{
  const double one = 1.0;

  HIP_CHECK(hipMemset(d_x, 0, precond_data.n * sizeof(double)));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(precond_data.handle_rocsparse,
                                         rocsparse_operation_none,
                                         precond_data.n,
                                         precond_data.nnz,
                                         &one,
                                         precond_data.descrL,
                                         precond_data.d_M_vals,
                                         precond_data.d_row_ptr,
                                         precond_data.d_col_idx,
                                         precond_data.infoM,
                                         d_r,
                                         precond_data.d_aux,
                                         rocsparse_solve_policy_auto,
                                         precond_data.buffer));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(precond_data.handle_rocsparse,
                                         rocsparse_operation_transpose,
                                         precond_data.n,
                                         precond_data.nnz,
                                         &one,
                                         precond_data.descrL,
                                         precond_data.d_M_vals,
                                         precond_data.d_row_ptr,
                                         precond_data.d_col_idx,
                                         precond_data.infoM,
                                         precond_data.d_aux,
                                         d_x,
                                         rocsparse_solve_policy_auto,
                                         precond_data.buffer));

  HIP_CHECK(hipDeviceSynchronize());
  return 0;
}

void cleanup_ic_preconditioner(PreconditionerData& precond_data)
{
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(precond_data.descrM));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(precond_data.descrL));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_info(precond_data.infoM));
  HIP_CHECK(hipFree(precond_data.buffer));
  HIP_CHECK(hipFree(precond_data.d_M_vals));
  HIP_CHECK(hipFree(precond_data.d_aux));
}
