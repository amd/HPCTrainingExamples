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
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_32x32x8f16 to
compute a 32x32x32 matrix multiplication.

Input:
  A : 32 x 32 float16s (a 32x32 matrix)
  B : 32 x 32 float16s (a 32x32 matrix)

Output:
  D : 32 x 32 floats (a 32x32 matrix)
*/

constexpr int M = 32;
constexpr int N = 32;
constexpr int K = 32;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;


__global__ void sgemm_32x32x32(const float16_t* A, const float16_t* B, float* D)
{

#if __gfx90a__ || __gfx908__
  // This kernel computes a 16x16x16 matrix multiplication using a single wavefront.
  using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  floatx16 d = {0}; // zero out 16 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_32x32x8f16 accumulates the sum of 8 outer products,
  8 columns of A with 8 rows of B, into result matrix D (which is in AccVGPRs).
  So we need four iterations to compute the full matrix D, starting with the leftmost 8
  columns of A and the topmost 8 colums of B, and then moving 8 columns to the right
  for A, and down 8 rows for B, for every iteration.

  For both the 8 columns of A, and the 8 rows of B, we use a single VGPR pair.
  With 64 lanes, and 4 Float16 values per lane, that covers the 8 columns of A and 8
  rows of B.
  Matrix A is a 32 x 8 matrix that is stored in 2 VGPRs as follows:
    first 32 lanes of a[0] cover column 0 -  last 32 lanes of a[0] cover column 4
    first 32 lanes of a[1] cover column 1 -  last 32 lanes of a[1] cover column 5
    first 32 lanes of a[2] cover column 2 -  last 32 lanes of a[2] cover column 6
    first 32 lanes of a[3] cover column 3 -  last 32 lanes of a[3] cover column 7
  Matrix B is a 8 x 32 matrix that is stored in 2 VGPRs as follows:
    first 32 lanes of b[0] cover row 0 -  last 32 lanes of b[0] cover row 4
    first 32 lanes of b[1] cover row 1 -  last 32 lanes of b[1] cover row 5
    first 32 lanes of b[2] cover row 2 -  last 32 lanes of b[2] cover row 6
    first 32 lanes of b[3] cover row 3 -  last 32 lanes of b[3] cover row 7
  Note that A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(32, 2) layout
  */

  for(int k = 0; k < 4; ++k){
    float16x4 a;
    float16x4 b;
    for(int i = 0; i < 4; ++i){
      const int a_idx =  threadIdx.x * LDA      // consecutive threads cover 32 consecutive rows
                       + i                      // consecutive registers take consecutive columns
                       + threadIdx.y * 4        // groups of 32 lanes skip 4 columns
                       + k * 8;                 // 8 columns fetched in each iteration
      a[i] = A[a_idx];

      const int b_idx =  threadIdx.x            // consecutive threads cover 32 consecutive columns
                       + i * LDB                // consecutive registers take consecutive rows
                       + threadIdx.y * 4 * LDB  // groups of 32 lanes skip 4 rows
                       + k * 8 * LDB;           // 8 rows fetched in each iteration
      b[i] = B[b_idx];
    }

    d = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, d, 0, 0, 0);
    //                                       ^  ^  ^
    //D(=C)                                  |  |  C(=D)
    //                      8 columns of A---|  |--- 8 rows of B
  }

  /*
  Matrix D is a 32 x 32 matrix that is stored in 16 AccVGPRs as follows:
    d[0:3]   cover rows 0-7
    d[4:7]   cover rows 8-15
    d[8:11]  cover rows 16-23
    d[12:15] cover rows 24-31
  Within each block of 4 AccVGPRs/8 rows (using d[0:3] as an example):
    first 32 lanes of d[0] cover row 0 -  last 32 lanes of d[0] cover row 4
    first 32 lanes of d[1] cover row 1 -  last 32 lanes of d[1] cover row 5
    first 32 lanes of d[2] cover row 2 -  last 32 lanes of d[2] cover row 6
    first 32 lanes of d[3] cover row 3 -  last 32 lanes of d[3] cover row 7
  */
  for(int j = 0; j < 4; ++j){
    for(int i = 0; i < 4; ++i){
      const int d_idx =  threadIdx.x            // consecutive threads cover 32 consecutive columns
                       + i * LDD                // consecutive registers take consecutive rows of 32 floats
                       + threadIdx.y * 4 * LDD  // last 32 lanes skip 4 rows
                       + j * 2 * 4 * LDD;       // blocks of 4 registers cover 8 rows

      D[d_idx] = d[i + 4 * j];
    }
  }
#endif
}


int main(){
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_f32_32x32x8f16 instruction only available on gfx908 or later."
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
  gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

  // Make and populate device buffers
  float16_t *A_d, *B_d;
  float *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float16_t)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float16_t), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  sgemm_32x32x32<<<1, dim3(32, 2)>>>(A_d, B_d, D_d);
  HIP_CHECK(hipGetLastError());

  // Copy result back to host
  std::vector<float> D_h(D_size);
  HIP_CHECK(hipMemcpy(D_h.data(), D_d, D_size * sizeof(float), hipMemcpyDeviceToHost));

  std::cout << "Sum of squared differences of host/device result matrices: "
            << compute_l2_error(Dref_h, D_h, M, N, LDD, LDD)
            << std::endl;

  HIP_CHECK(hipFree(D_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  return 0;
}
