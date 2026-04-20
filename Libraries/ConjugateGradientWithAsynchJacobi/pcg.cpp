#include "pcg.h"
#include <cstdio>
#include <cmath>

PCGResult pcg_solve(const CSRMatrix& A,
                    double* d_x,
                    const double* d_b,
                    int maxit,
                    double tol,
                    const std::string& precond_name,
                    PreconditionerData& precond_data,
                    rocblas_handle handle_rocblas,
                    rocsparse_handle handle_rocsparse)
{
  PCGResult result;
  result.iterations = 0;
  result.final_residual = 0.0;
  result.flag = -1;

  int nn = A.n;
  int nnnz = A.nnz;

  double* d_r;
  double* d_w;
  double* d_p;
  double* d_q;

  HIP_CHECK(hipMalloc(&d_r, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_w, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_p, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_q, sizeof(double) * nn));

  HIP_CHECK(hipMemset(d_r, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemset(d_w, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemset(d_p, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemset(d_q, 0, nn * sizeof(double)));

  double* h_res_norm_history = (double*) calloc(maxit + 2, sizeof(double));

  double alpha, beta, tolrel, rho_current, rho_previous, pTq;

  const double one = 1.0;
  const double zero = 0.0;
  const double minusone = -1.0;
  int notconv = 1, iter = 0;

  HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * nn, hipMemcpyDeviceToDevice));

  ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
                                   rocsparse_operation_none,
                                   nn,
                                   nn,
                                   nnnz,
                                   &minusone,
                                   A.descr,
                                   A.d_vals,
                                   A.d_row_ptr,
                                   A.d_col_idx,
                                   A.info,
                                   d_x,
                                   &one,
                                   d_r));

  ROCBLAS_CHECK(rocblas_ddot(handle_rocblas, nn, d_r, 1, d_r, 1, &h_res_norm_history[0]));

  h_res_norm_history[0] = sqrt(h_res_norm_history[0]);
  tolrel = tol * h_res_norm_history[0];

  printf("CG: it %d, res norm %5.5e \n", 0, h_res_norm_history[0]);

  while (notconv) {

    apply_preconditioner(precond_name, d_w, d_r, A, precond_data);

    ROCBLAS_CHECK(rocblas_ddot(handle_rocblas, nn, d_r, 1, d_w, 1, &rho_current));

    if (iter == 0) {
      HIP_CHECK(hipMemcpy(d_p, d_w, sizeof(double) * nn, hipMemcpyDeviceToDevice));
    } else {
      beta = rho_current / rho_previous;

      ROCBLAS_CHECK(rocblas_dscal(handle_rocblas, nn, &beta, d_p, 1));
      ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas, nn, &one, d_w, 1, d_p, 1));

      HIP_CHECK(hipDeviceSynchronize());
    }

    ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
                                     rocsparse_operation_none,
                                     nn,
                                     nn,
                                     nnnz,
                                     &one,
                                     A.descr,
                                     A.d_vals,
                                     A.d_row_ptr,
                                     A.d_col_idx,
                                     A.info,
                                     d_p,
                                     &zero,
                                     d_q));

    ROCBLAS_CHECK(rocblas_ddot(handle_rocblas, nn, d_p, 1, d_q, 1, &pTq));
    alpha = rho_current / pTq;

    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas, nn, &alpha, d_p, 1, d_x, 1));

    alpha *= (-1.0);
    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas, nn, &alpha, d_q, 1, d_r, 1));
    alpha *= (-1.0);

    iter++;
    ROCBLAS_CHECK(rocblas_ddot(handle_rocblas, nn, d_r, 1, d_r, 1, &h_res_norm_history[iter]));

    h_res_norm_history[iter] = sqrt(h_res_norm_history[iter]);

    printf("CG: it %d, res norm %5.16e \n", iter, h_res_norm_history[iter]);

    if ((h_res_norm_history[iter]) < tolrel) {
      result.flag = 0;
      notconv = 0;
      result.iterations = iter;

      HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * nn, hipMemcpyDeviceToDevice));
      ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
                                       rocsparse_operation_none,
                                       nn,
                                       nn,
                                       nnnz,
                                       &minusone,
                                       A.descr,
                                       A.d_vals,
                                       A.d_row_ptr,
                                       A.d_col_idx,
                                       A.info,
                                       d_x,
                                       &one,
                                       d_r));
      double r2_final;
      ROCBLAS_CHECK(rocblas_ddot(handle_rocblas, nn, d_r, 1, d_r, 1, &r2_final));
      printf("\nTRUE Norm of r %5.16e\n", sqrt(r2_final));

    } else {
      if (iter > maxit) {
        result.flag = 1;
        notconv = 0;
        result.iterations = iter;
      }
    }

    rho_previous = rho_current;
  }

  result.final_residual = h_res_norm_history[result.iterations];

  HIP_CHECK(hipFree(d_r));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_p));
  HIP_CHECK(hipFree(d_q));
  free(h_res_norm_history);

  return result;
}
