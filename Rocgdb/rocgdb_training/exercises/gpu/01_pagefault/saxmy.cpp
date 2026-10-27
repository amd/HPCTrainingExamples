// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// Topic: the saxmy page-fault demo for the rocgdb walkthrough (Part II).
//
// saxmy computes y = a*x*y element-wise (a scaled Hadamard product): deliberately
// NOT a BLAS routine, unlike axpy, so no one copies it in place of a library call.
// The fault is INJECTED here: the device pointers are never allocated (no hipMalloc),
// so the kernel dereferences null; hipDeviceSynchronize() lets the GPU report the
// fault before the host exits. The fix lives in saxmy_fixed.cpp.
// Please see README.md for building and running instructions.

#include <hip/hip_runtime.h>

__constant__ float a = 1.0f;

__global__
void saxmy(int n, float const* x, int incx, float* y, int incy)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] = a*x[i]*y[i];
}

int main()
{
    int n = 256;
    std::size_t size = sizeof(float)*n;

    float* d_x = nullptr;
    float* d_y = nullptr;

    int num_groups = 2;
    int group_size = 128;
    saxmy<<<num_groups, group_size>>>(n, d_x, 1, d_y, 1);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();        // HIP has no OMP_TARGET_OFFLOAD=MANDATORY;
    if (err != hipSuccess) {                   // on a node with no GPU the launch just
        fprintf(stderr, "HIP error: %s\n", hipGetErrorString(err));  // fails silently -
        return 1;                              // check it, or the demo "passes" with no fault
    }
}
