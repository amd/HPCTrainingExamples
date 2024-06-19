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
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_4x4x4f16 to
compute a batch of 16 4x4x4 matrix multiplications.

Input:
  A : 4 x 4 x 16 float16s (16 4x4 matrices)
  B : 4 x 4 x 16 float16s (16 4x4 matrices)

Output:
  D : 4 x 4 x 16 floats (16 4x4 matrices)
*/

constexpr int M = 4;
constexpr int N = 4;
constexpr int K = 4;
constexpr int nBatch = 16;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int batchStrideA = M * LDA;
constexpr int batchStrideB = K * LDB;
constexpr int batchStrideD = M * LDD;

constexpr int A_size = batchStrideA * nBatch;
constexpr int B_size = batchStrideB * nBatch;
constexpr int D_size = batchStrideD * nBatch;

__global__ void sgemm_4x4x4_batch(const float16_t *A, const float16_t *B, float *D)
{

#if __gfx90a__ || __gfx908__
  // This kernel computes a batch of 16 4x4x4 matrix multiplications using a single wavefront.
  using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  floatx4 d = {0}; // zero out 4 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_4x4x4f16 accumulates 16 batches of 4 outer products,
  four columns of each A with four rows of each B, into a batch of 16 result matrices D.
  So we therefore only need a single iteration to compute the full batch of 16 matrix
  multiplications

  For the four columns of each A, and the four rows of each B, we use a single VGPR pair.
  With 64 lanes, and 4 Float16 values per lane, that covers the 4 columns of each A and 4
  rows of each B.
  Matrix A is a batch of 16 4 x 4 matrices stored in 2 VGPRs as follows:
    lanes 0-3 contain the first A matrix, ..., lanes 60-63 contain the 16th A matrix
  Within a block of 4 lanes, e.g. lanes 0-3:
    a[0] covers column 0
    a[1] covers column 1
    a[2] covers column 2
    a[3] covers column 3
  Matrix B is a batch of 16 4 x 4 matrices stored in 2 VGPRs as follows:
    lanes 0-3 contain the first  A matrix, ..., lanes 60-63 contain the 16th A matrix
  Within a block of 4 lanes, e.g. lanes 0-3:
    b[0] covers row 0
    b[1] covers row 1
    b[2] covers row 2
    b[3] covers row 3
  Note that each A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(4, 16) layout
  */

  float16x4 a;
  float16x4 b;
  for(int i = 0; i < 4; ++i){
    const int a_idx =  threadIdx.x * LDA           // consecutive threads cover 16 consecutive rows
                     + i                           // consecutive registers take consecutive columns
                     + threadIdx.y * batchStrideA; // groups of 16 lanes cover each matrix in batch
    a[i] = A[a_idx];

    const int b_idx =  threadIdx.x                 // consecutive threads cover 16 consecutive columns
                     + i * LDB                     // consecutive registers take consecutive rows
                     + threadIdx.y * batchStrideB; // groups of 16 lanes cover each matrix in batch
    b[i] = B[b_idx];
  }

  d = __builtin_amdgcn_mfma_f32_4x4x4f16(a, b, d, 0, 0, 0);
  //                                     ^  ^  ^
  //D(=C)                                |  |  C(=D)
  //               4 columns of each A---|  |--- 4 rows of each B

  /*
  Matrix D is a batch of 16 4 x 4 matrices that are stored in 4 AccVGPRs as follows:
    d[0] covers row 0
    d[1] covers row 1
    d[2] covers row 2
    d[3] covers row 3
  */
  for (int i = 0; i < 4; ++i) {
    const int d_idx =   threadIdx.x                 // consecutive threads cover 4 consecutive columns
                      + i * LDD                     // consecutive registers take consecutive rows
                      + threadIdx.y * batchStrideD; // groups of 4 lanes cover each matrix in batch
    D[d_idx] = d[i];
  }
#endif
}


int main() {
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_f32_4x4x4f16 instruction only available on gfx908 or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<float16_t> A_h(A_size);
  for(int i = 0; i < A_h.size(); ++i){
    A_h[i] = static_cast<float16_t>(dist(gen));
  }
  std::vector<float16_t> B_h(B_size);
  for(int i = 0; i < B_h.size(); ++i){
    B_h[i] = static_cast<float16_t>(dist(gen));
  }

  // Calculate reference D on host
  std::vector<float> Dref_h(D_size);
  gemm_host_batch(A_h, B_h, Dref_h, M, N, K, nBatch,
                  LDA, LDB, LDD,
                  batchStrideA, batchStrideB, batchStrideD);

  // Make and populate device buffers
  float16_t *A_d, *B_d;
  float *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float16_t), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  sgemm_4x4x4_batch<<<1, dim3(4, 16)>>>(A_d, B_d, D_d);
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
