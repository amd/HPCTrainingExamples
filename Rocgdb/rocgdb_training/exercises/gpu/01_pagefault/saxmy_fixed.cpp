// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// Reference solution for 01_pagefault (diff this against saxmy.cpp).
//
// saxmy computes y = a*x*y element-wise (a scaled Hadamard product): deliberately
// NOT a BLAS routine, unlike axpy, so no one copies it in place of a library call.
// THE FIX: the two hipMalloc calls are added (the buggy saxmy.cpp leaves the device
// pointers null) and the device buffers are zero-initialised, so the kernel no longer
// dereferences null and the program runs clean.
// Build: amdclang++ -x hip -ggdb -O0 --offload-arch=gfx942 -o saxmy_fixed saxmy_fixed.cpp

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
    hipMalloc(&d_x, size);   // the fix: allocate device memory
    hipMalloc(&d_y, size);   // (and zero it below)
    hipMemset(d_x, 0, size);
    hipMemset(d_y, 0, size);

    int num_groups = 2;
    int group_size = 128;
    saxmy<<<num_groups, group_size>>>(n, d_x, 1, d_y, 1);
    hipDeviceSynchronize();

    hipFree(d_x);
    hipFree(d_y);
}
