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
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_16x16x1f32 to
compute a batch of four 16x16x16 matrix multiplications.

Input:
  A : 16 x 16 x 4 floats (four 16x16 matrices)
  B : 16 x 16 x 4 floats (four 16x16 matrices)

Output:
  D : 16 x 16 x 4 floats (four 16x16 matrices)
*/

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
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


__global__ void sgemm_16x16x16_batch(const float *A, const float *B, float *D)
{

#if __gfx90a__ || __gfx908__
  // This kernel computes a batch of four 16x16x16 matrix multiplications using a single wavefront.
  using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  float16 d = {0}; // zero out 16 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_16x16x1f32 calculates a batch of four outer products,
  one column of each A with one row of each B, into a batch of four result matrices D.
  So we need 16 iterations to compute the full batch of matrix multiplications,
  starting with the leftmost columns of each A and the topmost column of each B,
  and then moving one column to the right for each A, and down one row for each B,
  for every iteration.

  For the single column of each A, and the single row of each B, we use a single regular VGPR.
  With 64 lanes, that covers the 64 values for the four rows/columns of 16 items each.
  For the columns of the four A matrices: lanes 0-15 cover the column of the first matrix,
  ..., lanes 48-63 cover the column of the fourth matrix.
  For the rows of the four B matrices: lanes 0-15 cover the row of the first matrix,
  ..., lanes 48-63 cover the row of the fourth matrix.
  Note that each A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(16, 4) layout
  */

  int a_idx = LDA * threadIdx.x + batchStrideA * threadIdx.y;
  int b_idx = threadIdx.x + batchStrideB * threadIdx.y;

  for(int i = 0; i < 16; ++i){
    const float a = A[a_idx];
    const float b = B[b_idx];

    d = __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, d, 0, 0, 0);
    //                                       ^  ^  ^
    //D(=C)                                  |  |  C(=D)
    //              one column from each A---|  |--- one row from each B
    a_idx += 1;   // move one column to the right
    b_idx += LDB; // move one row down
  }

  /*
  Matrix D is a batch of four 16 x 16 matrices that are stored in 16 AccVGPRs as follows:
    d[0:3]     contain the first  D matrix
    d[4:7]     contain the second D matrix
    d[8:11]    contain the third  D matrix
    d[12:15]   contain the fourth D matrix
  Within a block of 4 AccVGPRs, e.g. d[0:3]:
    d[0] covers rows 0, 4, 8, and 12
    d[1] covers rows 1, 5, 9, and 13
    d[2] covers rows 2, 6, 10, and 14
    d[3] covers rows 3, 7, 11, and 15
    first 16 lanes of d[0] cover row 0 -  last 16 lanes of d[0] cover row 12
    first 16 lanes of d[1] cover row 1 -  last 16 lanes of d[1] cover row 13
    first 16 lanes of d[2] cover row 2 -  last 16 lanes of d[2] cover row 14
    first 16 lanes of d[3] cover row 3 -  last 16 lanes of d[3] cover row 15
  */
  for (int b = 0; b < 4; ++b) {
    for (int i = 0; i < 4; ++i) {
      const int d_idx =   threadIdx.x             // consecutive threads cover 16 consecutive columns
                        + i * LDD                 // consecutive registers take consecutive rows of 16 floats
                        + threadIdx.y * 4  * LDD  // groups of 16 lanes skip 4 rows
                        + b * batchStrideD;       // groups of 4 registers cover each matrix in batch
      D[d_idx] = d[i + b * 4];
    }
  }
#endif
}


int main() {
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_f32_16x16x1f32 instruction only available on gfx908 or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<float> A_h(A_size);
  for(int i = 0; i < A_h.size(); ++i){
    A_h[i] = dist(gen);
  }
  std::vector<float> B_h(B_size);
  for(int i = 0; i < B_h.size(); ++i){
    B_h[i] = dist(gen);
  }

  // Calculate reference D on host
  std::vector<float> Dref_h(D_size);
  gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch,
                  LDA, LDB, LDD,
                  batchStrideA, batchStrideB, batchStrideD);

  // Make and populate device buffers
  float *A_d, *B_d, *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  sgemm_16x16x16_batch<<<1, dim3(16, 4)>>>(A_d, B_d, D_d);
  HIP_CHECK(hipGetLastError());

  // Copy result back to host
  std::vector<float> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float), hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error_batch(Dref_h, D_h, M, N, nBatch,
                                      LDD, LDD, batchStrideD, batchStrideD)
            << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
