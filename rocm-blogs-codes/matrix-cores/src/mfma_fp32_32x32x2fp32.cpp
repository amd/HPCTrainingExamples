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
This example code uses the mfma intrinsic __builtin_amdgcn_mfma_f32_32x32x2f32 to
compute a 32x32x32 matrix multiplication.

Input:
  A : 32 x 32 floats (a 32x32 matrix)
  B : 32 x 32 floats (a 32x32 matrix)

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


__global__ void sgemm_32x32x32(const float* A, const float* B, float* D)
{

#if __gfx90a__ || __gfx908__
  // This kernel computes a 32x32x32 matrix multiplication using a single wavefront.
  using float16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  float16 d = {0}; // zero out 16 vanilla VGPRs

  /*
  One invocation of v_mfma_f32_32x32x2f32 accumulates the sum of two outer products,
  two columns of A with two rows of B, into result matrix D (which is in AccVGPRs).
  So we need 16 iterations to compute the full matrix D, starting with the leftmost two
  columns of A and the topmost two colums of B, and then moving two columns to the right
  for A, and down for B, for every iteration.

  For both the two columns of A, and the two rows of B, we use a single regular VGPR.
  With 64 lanes, that covers the 64 values for the two rows/columns of 32 items each.
  For the two A columns: lanes 0-31 cover the 1st column, lanes 32-63 cover the 2nd column.
  For the two B rows: lanes 0-31 cover the 1st row, lanes 32-63 cover the 2nd row.
  Note that A and B are in row-major order.

  This kernel is called with a single wavefront in dim3(32, 2) layout
  */

  int a_idx = LDA * threadIdx.x + threadIdx.y;
  int b_idx = threadIdx.x + LDB * threadIdx.y;

  for(int i = 0; i < 16; ++i){
    const float a = A[a_idx];
    const float b = B[b_idx];

    d = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, d, 0, 0, 0);
    //                                       ^  ^  ^
    //D(=C)                                  |  |  C(=D)
    //                    two columns of A---|  |--- two rows of B
    a_idx += 2;     // move two columns to the right
    b_idx += 2*LDB; // move two rows down
  }

  /*
  Matrix D is a 32 x 32 matrix that is stored in 16 AccVGPRs as follows:
    d[0:3]   cover rows 0-7 (256 floats)
    d[4:7]   cover rows 8-15
    d[8:11]  cover rows 16-23
    d[12:15] cover rows 24-31
  Within each block of 4 AccVGPRs/8 rows (using d[0:3] as an example:
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
    std::cout << "mfma_f32_32x32x2f32 instruction only available on gfx908 or later."
              << std::endl;
    exit(-1);
  }

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  // Make and populate some host matrices
  std::vector<float> A_h(A_size);
  for(int i = 0; i != A_h.size(); ++i){
    A_h[i] = dist(gen);
  }
  std::vector<float> B_h(B_size);
  for(int i = 0; i != B_h.size(); ++i){
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
