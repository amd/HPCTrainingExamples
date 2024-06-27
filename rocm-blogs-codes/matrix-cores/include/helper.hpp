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

#ifndef _HELPER_HPP_
#define _HELPER_HPP_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>

using float16_t = _Float16;

// HIP error check
#define HIP_CHECK(command)                                    \
{                                                             \
  hipError_t stat = (command);                                \
  if(stat != hipSuccess)                                      \
  {                                                           \
    std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
    " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
    exit(-1);                                                 \
  }                                                           \
}

bool gpuArchCheck(const std::string arch) {
  hipDeviceProp_t hipProps;
  int deviceId=0;
  HIP_CHECK(hipGetDevice(&deviceId));
  HIP_CHECK(hipGetDeviceProperties(&hipProps, deviceId));
  return (std::string(hipProps.gcnArchName).find(arch)
          !=std::string::npos);
}

std::ostream& operator<<(std::ostream& os, const float16_t& val) {
    os << static_cast<float>(val);
    return os;
}

template<typename T>
void print_matrix(const std::vector<T>& A,
                  const int M,
                  const int N,
                  const int LDA) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      std::cout << A[n + m * LDA] << "  ";
    }
    std::cout << std::endl;
  }
}

template<typename T>
void print_matrix_batch(const std::vector<T>& A,
                        const int M,
                        const int N,
                        const int nBatch,
                        const int LDA,
                        const int batchStride) {
  for (int b = 0; b < nBatch; ++b) {
    std::cout << "Batch " << b << ":" << std::endl;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        std::cout << A[n + m * LDA + b * batchStride] << "  ";
      }
      std::cout << std::endl;
    }
  }
}

template<typename T, typename U>
void gemm_host(const std::vector<U>& A,
               const std::vector<U>& B,
                     std::vector<T>& C,
               const int M,
               const int N,
               const int K,
               const int LDA,
               const int LDB,
               const int LDC) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      T c = 0.0;
      for (int k = 0; k < K; ++k) {
        c += A[k + m * LDA] * B[n + k * LDB];
      }
      C[n + m * LDC] = c;
    }
  }
}

template<typename T, typename U>
void gemm_host_batch(const std::vector<U>& A,
                     const std::vector<U>& B,
                           std::vector<T>& C,
                     const int M,
                     const int N,
                     const int K,
                     const int nBatch,
                     const int LDA,
                     const int LDB,
                     const int LDC,
                     const int batchStrideA,
                     const int batchStrideB,
                     const int batchStrideC) {
  for (int b = 0; b < nBatch; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        T c = 0.0;
        for (int k = 0; k < K; ++k) {
          c += A[k + m * LDA + b * batchStrideA] * B[n + k * LDB + b * batchStrideB];
        }
        C[n + m * LDC + b * batchStrideC] = c;
      }
    }
  }
}

template<typename T>
double compute_l2_error(const std::vector<T>& A,
                        const std::vector<T>& B,
                        const int M,
                        const int N,
                        const int LDA,
                        const int LDB) {

  double err = 0.0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      const double x = A[n + LDA * m] - B[n + LDB * m];
      err += x * x;
    }
  }
  return err;
}

template<typename T>
double compute_l2_error_batch(const std::vector<T>& A,
                              const std::vector<T>& B,
                              const int M,
                              const int N,
                              const int nBatch,
                              const int LDA,
                              const int LDB,
                              const int batchStrideA,
                              const int batchStrideB) {

  double err = 0.0;
  for (int b = 0; b < nBatch; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        const double x = A[n + LDA * m + b * batchStrideA] - B[n + LDB * m + b * batchStrideB];
        err += x * x;
      }
    }
  }
  return err;
}

#endif
