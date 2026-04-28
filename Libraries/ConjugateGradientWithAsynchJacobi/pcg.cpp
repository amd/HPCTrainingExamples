#include "pcg.h"
#include <cstdio>
#include <cmath>

__global__ void compute_alpha_kernel(double* d_alpha, const double* d_rho, const double* d_pTq)
{
  d_alpha[0] = d_rho[0] / d_pTq[0];
}

__global__ void compute_beta_kernel(double* d_beta, const double* d_rho_current, const double* d_rho_previous)
{
  d_beta[0] = d_rho_current[0] / d_rho_previous[0];
}

__global__ void negate_kernel(double* d_val)
{
  d_val[0] = -d_val[0];
}

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

  // Create dense vector descriptors for spmv
  rocsparse_dnvec_descr vec_x, vec_r, vec_p, vec_q;
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x,
                                               nn,
                                               d_x,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_r,
                                               nn,
                                               d_r,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_p,
                                               nn,
                                               d_p,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_q,
                                               nn,
                                               d_q,
                                               rocsparse_datatype_f64_r));

  // Device scalars
  double* d_one;
  double* d_zero;
  double* d_minusone;
  double* d_alpha;
  double* d_beta;
  double* d_rho_current;
  double* d_rho_previous;
  double* d_pTq;
  double* d_res_norm_sq;

  HIP_CHECK(hipMalloc(&d_one, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_zero, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_minusone, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_alpha, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_beta, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_rho_current, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_rho_previous, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_pTq, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_res_norm_sq, sizeof(double)));

  double h_one = 1.0, h_zero = 0.0, h_minusone = -1.0;
  HIP_CHECK(hipMemcpy(d_one, &h_one, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_zero, &h_zero, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_minusone, &h_minusone, sizeof(double), hipMemcpyHostToDevice));

  double h_res_norm_sq, tolrel;
  int notconv = 1, iter = 0;
  rocsparse_error spmv_error;

  // r = b - A*x
  HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * nn, hipMemcpyDeviceToDevice));
  ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                    A.spmv_descr,
                                    d_minusone,
                                    A.spmat,
                                    vec_x,
                                    d_one,
                                    vec_r,
                                    rocsparse_v2_spmv_stage_compute,
                                    A.spmv_buffer_size,
                                    A.spmv_buffer,
                                    &spmv_error));

  ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                             nn,
                             d_r,
                             1,
                             d_r,
                             1,
                             d_res_norm_sq));
  HIP_CHECK(hipMemcpy(&h_res_norm_sq, d_res_norm_sq, sizeof(double), hipMemcpyDeviceToHost));

  double h_res_norm = sqrt(h_res_norm_sq);
  tolrel = tol * h_res_norm;

  printf("CG: it %d, res norm %5.5e \n", 0, h_res_norm);

  while (notconv) {

    apply_preconditioner(precond_name, d_w, d_r, A, precond_data);

    ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                               nn,
                               d_r,
                               1,
                               d_w,
                               1,
                               d_rho_current));

    if (iter == 0) {
      HIP_CHECK(hipMemcpy(d_p, d_w, sizeof(double) * nn, hipMemcpyDeviceToDevice));
    } else {
      compute_beta_kernel<<<1, 1>>>(d_beta, d_rho_current, d_rho_previous);

      ROCBLAS_CHECK(rocblas_dscal(handle_rocblas,
                                  nn,
                                  d_beta,
                                  d_p,
                                  1));
      ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                                  nn,
                                  d_one,
                                  d_w,
                                  1,
                                  d_p,
                                  1));

      HIP_CHECK(hipDeviceSynchronize());
    }

    // q = A*p
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                      A.spmv_descr,
                                      d_one,
                                      A.spmat,
                                      vec_p,
                                      d_zero,
                                      vec_q,
                                      rocsparse_v2_spmv_stage_compute,
                                      A.spmv_buffer_size,
                                      A.spmv_buffer,
                                      &spmv_error));

    ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                               nn,
                               d_p,
                               1,
                               d_q,
                               1,
                               d_pTq));

    compute_alpha_kernel<<<1, 1>>>(d_alpha, d_rho_current, d_pTq);

    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                                nn,
                                d_alpha,
                                d_p,
                                1,
                                d_x,
                                1));

    negate_kernel<<<1, 1>>>(d_alpha);
    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                                nn,
                                d_alpha,
                                d_q,
                                1,
                                d_r,
                                1));

    // Save rho_current to rho_previous
    HIP_CHECK(hipMemcpy(d_rho_previous, d_rho_current, sizeof(double), hipMemcpyDeviceToDevice));

    iter++;
    ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                               nn,
                               d_r,
                               1,
                               d_r,
                               1,
                               d_res_norm_sq));

    HIP_CHECK(hipMemcpy(&h_res_norm_sq, d_res_norm_sq, sizeof(double), hipMemcpyDeviceToHost));
    h_res_norm = sqrt(h_res_norm_sq);

    printf("CG: it %d, res norm %5.16e \n", iter, h_res_norm);

    if (h_res_norm < tolrel) {
      result.flag = 0;
      notconv = 0;
      result.iterations = iter;
      result.final_residual = h_res_norm;

      // Compute true residual: r = b - A*x
      HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * nn, hipMemcpyDeviceToDevice));
      ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                        A.spmv_descr,
                                        d_minusone,
                                        A.spmat,
                                        vec_x,
                                        d_one,
                                        vec_r,
                                        rocsparse_v2_spmv_stage_compute,
                                        A.spmv_buffer_size,
                                        A.spmv_buffer,
                                        &spmv_error));
      ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                                 nn,
                                 d_r,
                                 1,
                                 d_r,
                                 1,
                                 d_res_norm_sq));
      HIP_CHECK(hipMemcpy(&h_res_norm_sq, d_res_norm_sq, sizeof(double), hipMemcpyDeviceToHost));
      printf("\nTRUE Norm of r %5.16e\n", sqrt(h_res_norm_sq));

    } else {
      if (iter > maxit) {
        result.flag = 1;
        notconv = 0;
        result.iterations = iter;
        result.final_residual = h_res_norm;
      }
    }
  }

  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_r));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_p));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_q));

  HIP_CHECK(hipFree(d_r));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_p));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_one));
  HIP_CHECK(hipFree(d_zero));
  HIP_CHECK(hipFree(d_minusone));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_rho_current));
  HIP_CHECK(hipFree(d_rho_previous));
  HIP_CHECK(hipFree(d_pTq));
  HIP_CHECK(hipFree(d_res_norm_sq));

  return result;
}
