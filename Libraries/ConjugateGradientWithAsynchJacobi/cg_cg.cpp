#include "cg_cg.h"
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 1024
#define WARP_SIZE 64

__global__ void fused_dot_products_kernel(int n,
                                          const double* d_r,
                                          const double* d_w,
                                          const double* d_u,
                                          double* d_aux)
{
  __shared__ double s_rTu[BLOCK_SIZE / WARP_SIZE];
  __shared__ double s_wTu[BLOCK_SIZE / WARP_SIZE];
  __shared__ double s_rTr[BLOCK_SIZE / WARP_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int lane = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  double rTu = 0.0, wTu = 0.0, rTr = 0.0;

  while (i < n) {
    double u = d_u[i];
    double r = d_r[i];
    double w = d_w[i];
    rTu += r * u;
    wTu += w * u;
    rTr += r * r;
    i += stride;
  }

#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    rTu += __shfl_down(rTu, offset, WARP_SIZE);
    wTu += __shfl_down(wTu, offset, WARP_SIZE);
    rTr += __shfl_down(rTr, offset, WARP_SIZE);
  }

  if (lane == 0) {
    s_rTu[warp_id] = rTu;
    s_wTu[warp_id] = wTu;
    s_rTr[warp_id] = rTr;
  }
  __syncthreads();

  int num_warps = BLOCK_SIZE / WARP_SIZE;
  if (tid < num_warps) {
    rTu = s_rTu[tid];
    wTu = s_wTu[tid];
    rTr = s_rTr[tid];

#pragma unroll
    for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
      rTu += __shfl_down(rTu, offset, WARP_SIZE);
      wTu += __shfl_down(wTu, offset, WARP_SIZE);
      rTr += __shfl_down(rTr, offset, WARP_SIZE);
    }

    if (tid == 0) {
      atomicAdd(&d_aux[0], rTu);
      atomicAdd(&d_aux[1], wTu);
      atomicAdd(&d_aux[2], rTr);
    }
  }
}

__global__ void compute_alpha_cgcg_kernel(double* d_alpha,
                                          const double* d_gamma_prev,
                                          const double* d_wu)
{
  d_alpha[0] = d_gamma_prev[0] / d_wu[0];
}

CGCGResult cg_cg_solve(const CSRMatrix& A,
                       double* d_x,
                       const double* d_b,
                       int maxit,
                       double tol,
                       const std::string& precond_name,
                       PreconditionerData& precond_data,
                       rocblas_handle handle_rocblas,
                       rocsparse_handle handle_rocsparse,
                       int low_synch)
{
  CGCGResult result;
  result.iterations = 0;
  result.final_residual = 0.0;
  result.flag = -1;

  int n = A.n;

  double* d_r;
  double* d_u;
  double* d_p;
  double* d_s;
  double* d_w;
  double* d_aux = nullptr;

  HIP_CHECK(hipMalloc(&d_r, sizeof(double) * n));
  HIP_CHECK(hipMalloc(&d_u, sizeof(double) * n));
  HIP_CHECK(hipMalloc(&d_p, sizeof(double) * n));
  HIP_CHECK(hipMalloc(&d_s, sizeof(double) * n));
  HIP_CHECK(hipMalloc(&d_w, sizeof(double) * n));

  if (low_synch == 1) {
    HIP_CHECK(hipMalloc(&d_aux, 3 * sizeof(double)));
  }

  HIP_CHECK(hipMemset(d_p, 0, n * sizeof(double)));
  HIP_CHECK(hipMemset(d_s, 0, n * sizeof(double)));

  // Create dense vector descriptors for spmv
  rocsparse_dnvec_descr vec_x, vec_r, vec_u, vec_w;
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x,
                                               n,
                                               d_x,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_r,
                                               n,
                                               d_r,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_u,
                                               n,
                                               d_u,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_w,
                                               n,
                                               d_w,
                                               rocsparse_datatype_f64_r));

  // Device scalars
  double* d_one;
  double* d_zero;
  double* d_minusone;
  double* d_alpha;
  double* d_neg_alpha;
  double* d_beta;
  double* d_gamma;
  double* d_gamma_prev;
  double* d_delta;
  double* d_res;
  double* d_res0;
  double* d_wu;

  HIP_CHECK(hipMalloc(&d_one, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_zero, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_minusone, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_alpha, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_neg_alpha, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_beta, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_gamma, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_gamma_prev, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_delta, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_res, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_res0, sizeof(double)));
  HIP_CHECK(hipMalloc(&d_wu, sizeof(double)));

  double h_one = 1.0, h_zero = 0.0, h_minusone = -1.0;
  HIP_CHECK(hipMemcpy(d_one, &h_one, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_zero, &h_zero, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_minusone, &h_minusone, sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta, &h_zero, sizeof(double), hipMemcpyHostToDevice));

  double h_res, h_res0, h_gamma, h_delta, h_alpha;
  rocsparse_error spmv_error;

  // r = b - A*x
  HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));
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

  // u = precond(r)
  apply_preconditioner(precond_name, d_u, d_r, A, precond_data);

  // w = A * u
  ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                    A.spmv_descr,
                                    d_one,
                                    A.spmat,
                                    vec_u,
                                    d_zero,
                                    vec_w,
                                    rocsparse_v2_spmv_stage_compute,
                                    A.spmv_buffer_size,
                                    A.spmv_buffer,
                                    &spmv_error));

  // gamma_prev = r' * u
  ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                             n,
                             d_r,
                             1,
                             d_u,
                             1,
                             d_gamma_prev));

  // res0 = r' * r
  ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                             n,
                             d_r,
                             1,
                             d_r,
                             1,
                             d_res0));
  HIP_CHECK(hipMemcpy(&h_res0, d_res0, sizeof(double), hipMemcpyDeviceToHost));

  // wu = w' * u
  ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                             n,
                             d_w,
                             1,
                             d_u,
                             1,
                             d_wu));

  // alpha = gamma_prev / wu
  compute_alpha_cgcg_kernel<<<1, 1>>>(d_alpha, d_gamma_prev, d_wu);

  printf("CG-CG: it %d, res norm %5.5e \n", 0, sqrt(h_res0));

  int notconv = 1;
  int iter = 0;

  while (notconv) {
    // p = u + beta * p
    ROCBLAS_CHECK(rocblas_dscal(handle_rocblas,
                                n,
                                d_beta,
                                d_p,
                                1));
    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                                n,
                                d_one,
                                d_u,
                                1,
                                d_p,
                                1));

    // s = w + beta * s
    ROCBLAS_CHECK(rocblas_dscal(handle_rocblas,
                                n,
                                d_beta,
                                d_s,
                                1));
    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                                n,
                                d_one,
                                d_w,
                                1,
                                d_s,
                                1));

    // x = x + alpha * p
    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                                n,
                                d_alpha,
                                d_p,
                                1,
                                d_x,
                                1));

    // r = r - alpha * s
    HIP_CHECK(hipMemcpy(&h_alpha, d_alpha, sizeof(double), hipMemcpyDeviceToHost));
    double h_neg_alpha = -h_alpha;
    HIP_CHECK(hipMemcpy(d_neg_alpha, &h_neg_alpha, sizeof(double), hipMemcpyHostToDevice));
    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas,
                                n,
                                d_neg_alpha,
                                d_s,
                                1,
                                d_r,
                                1));

    // u = precond(r)
    apply_preconditioner(precond_name, d_u, d_r, A, precond_data);

    // w = A * u
    ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                      A.spmv_descr,
                                      d_one,
                                      A.spmat,
                                      vec_u,
                                      d_zero,
                                      vec_w,
                                      rocsparse_v2_spmv_stage_compute,
                                      A.spmv_buffer_size,
                                      A.spmv_buffer,
                                      &spmv_error));

    if (low_synch == 0) {
      // gamma = r' * u
      ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                                 n,
                                 d_r,
                                 1,
                                 d_u,
                                 1,
                                 d_gamma));

      // delta = w' * u
      ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                                 n,
                                 d_w,
                                 1,
                                 d_u,
                                 1,
                                 d_delta));

      // res = r' * r
      ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                                 n,
                                 d_r,
                                 1,
                                 d_r,
                                 1,
                                 d_res));

      HIP_CHECK(hipMemcpy(&h_gamma, d_gamma, sizeof(double), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&h_delta, d_delta, sizeof(double), hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&h_res, d_res, sizeof(double), hipMemcpyDeviceToHost));
    } else {
      int block_size = BLOCK_SIZE;
      int num_blocks = (n + block_size - 1) / block_size;
      HIP_CHECK(hipMemset(d_aux, 0, 3 * sizeof(double)));
      fused_dot_products_kernel<<<num_blocks, block_size>>>(n, d_r, d_w, d_u, d_aux);
      HIP_CHECK(hipDeviceSynchronize());

      double h_aux[3];
      HIP_CHECK(hipMemcpy(h_aux, d_aux, 3 * sizeof(double), hipMemcpyDeviceToHost));
      h_gamma = h_aux[0];
      h_delta = h_aux[1];
      h_res = h_aux[2];

      HIP_CHECK(hipMemcpy(d_gamma, &h_gamma, sizeof(double), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_delta, &h_delta, sizeof(double), hipMemcpyHostToDevice));
    }

    iter++;
    printf("CG-CG: it %d, res norm %5.16e, scaled %5.16e\n", iter, sqrt(h_res), sqrt(h_res / h_res0));

    // beta = gamma / gamma_prev
    double h_gamma_prev;
    HIP_CHECK(hipMemcpy(&h_gamma_prev, d_gamma_prev, sizeof(double), hipMemcpyDeviceToHost));
    double h_beta = h_gamma / h_gamma_prev;
    HIP_CHECK(hipMemcpy(d_beta, &h_beta, sizeof(double), hipMemcpyHostToDevice));

    // alpha = (gamma * alpha_old) / (delta * alpha_old - beta * gamma)
    double denom = h_delta * h_alpha - h_beta * h_gamma;
    h_alpha = (h_gamma * h_alpha) / denom;
    HIP_CHECK(hipMemcpy(d_alpha, &h_alpha, sizeof(double), hipMemcpyHostToDevice));

    // gamma_prev = gamma
    HIP_CHECK(hipMemcpy(d_gamma_prev, d_gamma, sizeof(double), hipMemcpyDeviceToDevice));

    // Check convergence
    if (h_res / h_res0 < tol * tol) {
      result.flag = 0;
      notconv = 0;
      result.iterations = iter;
      result.final_residual = sqrt(h_res);

      // Compute true residual
      HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * n, hipMemcpyDeviceToDevice));
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
      double h_true_res;
      ROCBLAS_CHECK(rocblas_ddot(handle_rocblas,
                                 n,
                                 d_r,
                                 1,
                                 d_r,
                                 1,
                                 d_res));
      HIP_CHECK(hipMemcpy(&h_true_res, d_res, sizeof(double), hipMemcpyDeviceToHost));
      printf("\nTRUE Norm of r %5.16e\n", sqrt(h_true_res));
    } else if (iter >= maxit) {
      result.flag = 1;
      notconv = 0;
      result.iterations = iter;
      result.final_residual = sqrt(h_res);
    }
  }

  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_x));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_r));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_u));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vec_w));

  HIP_CHECK(hipFree(d_r));
  HIP_CHECK(hipFree(d_u));
  HIP_CHECK(hipFree(d_p));
  HIP_CHECK(hipFree(d_s));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_one));
  HIP_CHECK(hipFree(d_zero));
  HIP_CHECK(hipFree(d_minusone));
  HIP_CHECK(hipFree(d_alpha));
  HIP_CHECK(hipFree(d_neg_alpha));
  HIP_CHECK(hipFree(d_beta));
  HIP_CHECK(hipFree(d_gamma));
  HIP_CHECK(hipFree(d_gamma_prev));
  HIP_CHECK(hipFree(d_delta));
  HIP_CHECK(hipFree(d_res));
  HIP_CHECK(hipFree(d_res0));
  HIP_CHECK(hipFree(d_wu));
  if (d_aux != nullptr) {
    HIP_CHECK(hipFree(d_aux));
  }

  return result;
}
