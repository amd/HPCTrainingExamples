/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "dgemm.h"
#include "darray.h"
#include "timer.h"



dgemm_results
run_dgemm(matrixd const& A, matrixd const& B,
          int iter_count, int rep_count, int dev_id){

   hipSetDevice(dev_id);
   hipblasHandle_t handle;
   hipblasCreate(&handle);

   // Matrix dimensions for C = A * B
   // A is (m x n), B is (n x k), C is (m x k)
   int const m = A.row_count();
   int const n = A.col_count();
   int const k = B.col_count();

   matrixd C(m, k);

   darray<double> A_mat(m*n);
   darray<double> B_mat(n*k);
   darray<double> C_mat(m*k);

   double const alpha_val = 1.0;
   double const beta_val = 1.0;

   std::vector<double> mat_zeros(m*k);
   hipMemcpy(A_mat.data(), &A.data()[0], m*n*sizeof(double), hipMemcpyHostToDevice);
   hipMemcpy(B_mat.data(), &B.data()[0], n*k*sizeof(double), hipMemcpyHostToDevice);
   hipMemcpy(C_mat.data(), &mat_zeros[0], m*k*sizeof(double), hipMemcpyHostToDevice);

   dgemm_results ret;

   // hipblasDgemm uses column-major storage, but our matrices are row-major.
   // To compute C = A * B with row-major matrices using column-major BLAS:
   //   - Row-major matrix X is seen as X^T by column-major BLAS
   //   - We use the identity: C = A * B  <=>  C^T = B^T * A^T
   //   - So we call hipblasDgemm with B and A swapped (no transpose flags)
   //   - The result C^T in column-major is C in row-major
   //
   // hipblasDgemm computes: C_blas = alpha * op(A_blas) * op(B_blas) + beta * C_blas
   // We call: C_blas = alpha * B_rowmajor * A_rowmajor + beta * C_blas
   // BLAS sees: C^T = alpha * B^T * A^T + beta * C^T = alpha * (A*B)^T + beta * C^T
   // Reading C^T as row-major gives us C = alpha * A * B + beta * C

   for (int batch_idx=0; batch_idx<iter_count+1; ++batch_idx){
     timer t;
     t.tick();
     for (int i=0; i<rep_count; ++i){
       auto stat = hipblasDgemm(
          handle,
          HIPBLAS_OP_N, HIPBLAS_OP_N,
          k, m, n,
          &alpha_val, B_mat.data(), k,
          A_mat.data(), n,
          &beta_val, C_mat.data(), k);
     }
     check_stat(hipDeviceSynchronize());

     // Ignore the first set
     if (batch_idx == 0){
       continue;
     }

     double const dur_ms = t.tock<std::chrono::duration<double, std::milli>>();
     double const op_count = rep_count*2.0*m*n*k;
     // ops/ms * 1000ms/1s * 1 gops/1e9 * 1 tops/1e3gops  ops
     double const tflops = op_count/dur_ms/1.0e9;
     ret.flops.push_back(tflops);
     ret.time_points.push_back(now_str());
     printf("%d   %3d   %4.2f\n", dev_id, batch_idx, tflops);
   }

   return ret;
}
