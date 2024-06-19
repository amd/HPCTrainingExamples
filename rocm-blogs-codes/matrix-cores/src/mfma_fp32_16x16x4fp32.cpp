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
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_16x16x4f32 to
compute a 16x16x16 matrix multiplication.

Input:
  A : 16 x 16 floats (a 16x16 matrix)
  B : 16 x 16 floats (a 16x16 matrix)

Output:
  D : 16 x 16 floats (a 16x16 matrix)
*/

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDD = N;

constexpr int A_size = M * LDA;
constexpr int B_size = K * LDB;
constexpr int D_size = M * LDD;


__global__ void sgemm_16x16x16(const float* A, const float* B, float* D)
{

#if __gfx90a__ || __gfx908__
  // This kernel computes a 16x16x16 matrix multiplication using a single wavefront.
  using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float4 d = {0}; // zero out 4 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_16x16x4f32 accumulates the sum of four outer products,
  four columns of A with four rows of B, into result matrix D (which is in AccVGPRs).
  So we need 4 iterations to compute the full matrix D, starting with the leftmost four
  columns of A and the topmost four colums of B, and then moving four columns to the right
  for A, and down for B, for every iteration.

  For both the four columns of A, and the four rows of B, we use a single regular VGPR.
  With 64 lanes, that covers the 64 values for the four rows/columns of 16 items each.
  For the four A columns: lanes 0-15 cover the 1st column, ..., lanes 48-63 cover the 4th column.
  For the four B rows: lanes 0-15 cover the 1st row, ..., lanes 48-63 cover the 4th row.
  Note that A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(16, 4) layout
  */

  int a_idx = LDA * threadIdx.x + threadIdx.y;
  int b_idx = threadIdx.x + LDB * threadIdx.y;

  for(int i = 0; i < 4; ++i){
    float a = A[a_idx];
    float b = B[b_idx];

    d = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, d, 0, 0, 0);
    //                                       ^  ^  ^
    //D(=C)                                  |  |  C(=D)
    //                   four columns of A---|  |--- four rows of B
    a_idx += 4;     // move four columns to the right
    b_idx += 4*LDB; // move four rows down
  }

  /*
  Matrix D is a 16 x 16 matrix that is stored in 4 AccVGPRs as follows:
    d[0] covers rows 0, 4, 8, and 12
    d[1] covers rows 1, 5, 9, and 13
    d[2] covers rows 2, 6, 10, and 14
    d[3] covers rows 3, 7, 11, and 15
    first 16 lanes of d[0] cover row 0 -  last 16 lanes of d[0] cover row 12
    first 16 lanes of d[1] cover row 1 -  last 16 lanes of d[1] cover row 13
    first 16 lanes of d[2] cover row 2 -  last 16 lanes of d[2] cover row 14
    first 16 lanes of d[3] cover row 3 -  last 16 lanes of d[3] cover row 15
  */
  for(int i = 0; i < 4; ++i){
    const int d_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                     + i * LDD                // consecutive registers take consecutive rows of 16 floats
                     + threadIdx.y * 4 * LDD; // groups of 16 lanes skip 4 rows

    D[d_idx] = d[i];
  }
#endif
}


int main(){
  if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
    std::cout << "mfma_f32_16x16x4f32 instruction only available on gfx908 or later."
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
  gemm_host(A_h, B_h, Dref_h, M, N, K, LDA, LDB, LDD);

  // Make and populate device buffers
  float *A_d, *B_d, *D_d;
  HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&D_d, D_size * sizeof(float)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), A_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), B_size * sizeof(float), hipMemcpyHostToDevice));

  // Launch GEMM kernel
  sgemm_16x16x16<<<1, dim3(16, 4)>>>(A_d, B_d, D_d);
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
