/*
Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include "helper.hpp"

/*
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f64_4x4x4f64 to
compute a batch of 4 4x4x4 matrix multiplications.

Input:
  A : 4 x 4 x 4 doubles (four 4x4 matrices)
  B : 4 x 4 x 4 doubles (four 4x4 matrices)

Output:
  D : 4 x 4 x 4 doubles (four 4x4 matrices)
*/

constexpr int M = 4;
constexpr int N = 4;
constexpr int K = 4;
constexpr int nBatch = 4;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int batchStrideA = M * LDA;
constexpr int batchStrideB = K * LDB;
constexpr int batchStrideD = M * LDD;

constexpr int A_size = batchStrideA * nBatch;
constexpr int B_size = batchStrideB * nBatch;
constexpr int D_size = batchStrideD * nBatch;

__global__ void dgemm_4x4x4_batch(const double *A, const double *B, double *D)
{

#if __gfx90a__
  // This kernel computes a batch of four 4x4x4 matrix multiplications using a single wavefront.
  double d = {0}; // zero out 1 * 2 vanilla VGPRs

  /*
  One invocation of v_mfma_f64_4x4x4f64 accumulates four batches of four outer products,
  four columns of each A with four rows of each B, into a batch of four result matrices D.
  So we therefore only need a single iteration to compute the full batch of four matrix
  multiplications

  For the four columns of each A, and the four rows of each B, we use a single VGPR pair.
  With 64 lanes, this covers the 64 values for the four batches of 16 matrix entries.
  For the columns of the four A matrices: lanes 0-15 cover the first column of each of the
  four matrices (with lanes 0-3 being the first column of the first matrix, ..., lanes
  12-15 being the first column of the fourth matrix), ..., lanes 48-63 cover the forth
  column of each of the four matrices.
  For the rows of the four B matrices: lanes 0-15 cover the first row of each of the
  four matrices (with lanes 0-3 being the first row of the first matrix, ..., lanes
  12-15 being the first row of the fourth matrix), ..., lanes 48-63 cover the forth
  row of each of the four matrices.
  Note that each A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(4, 4, 4) layout
  */

  int a_idx = LDA * threadIdx.x + threadIdx.z + batchStrideA * threadIdx.y;
  int b_idx = threadIdx.x + LDB * threadIdx.z + batchStrideB * threadIdx.y;

  const double a = A[a_idx];
  const double b = B[b_idx];

  d = __builtin_amdgcn_mfma_f64_4x4x4f64(a, b, d, 0, 0, 0);
  //                                     ^  ^  ^
  //D(=C)                                |  |  C(=D)
  //            one column from each A---|  |--- one row from each B

  /*
  Matrix D is a batch of four 4 x 4 matrices that are stored in 1 AccVGPR pair as follows:
    lanes 0-15  cover the first row of each of the four D matrices
    lanes 16-31 cover the second row of each of the four D matrices
    lanes 32-47 cover the third row of each of the four D matrices
    lanes 48-63 cover the fourth row of each of the four D matrices
  Within each set of 16 lanes (using lanes 0-15 as an example):
    lanes 0-3   cover the first row of the first D matrix
    lanes 4-7   cover the first row of the second D matrix
    lanes 8-11  cover the first row of the third D matrix
    lanes 11-15 cover the first row of the fourth D matrix
  */
  const int d_idx =   threadIdx.x                 // consecutive threads cover 4 consecutive columns
                    + threadIdx.y * batchStrideD  // groups of 4 lanes cover a row of each matrix in batch
                    + threadIdx.z * LDD;          // groups of 16 lanes take consecutive rows
  D[d_idx] = d;
#endif
}


int main() {
  if (!gpuArchCheck("gfx90a")) {
    std::cout << "mfma_f64_4x4x4f64 instruction only available on gfx90a or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<double> A_h(A_size);
  for(int i = 0; i < A_h.size(); ++i){
    A_h[i] = dist(gen);
  }
  std::vector<double> B_h(B_size);
  for(int i = 0; i < B_h.size(); ++i){
    B_h[i] = dist(gen);
  }

  // Calculate reference D on host
  std::vector<double> Dref_h(D_size);
  gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch,
                  LDA, LDB, LDD,
                  batchStrideA, batchStrideB, batchStrideD);

  // Make and populate device buffers
  double *A_d, *B_d, *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(double)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(double)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(double), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  dgemm_4x4x4_batch<<<1, dim3(4, 4, 4)>>>(A_d, B_d, D_d);
  HIP_CHECK(hipGetLastError());

  // Copy result back to host
  std::vector<double> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(double), hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error_batch(Dref_h, D_h, M, N, nBatch,
                                      LDD, LDD, batchStrideD, batchStrideD)
            << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
