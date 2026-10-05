// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// Reference solution for 03_mpi_pagefault (diff this against mpi_saxmy.cpp).
// THE FIX: every rank now allocates y (the `if (rank != size - 1)` guard is gone),
// so no rank passes a null device pointer and every rank finishes cleanly.
//
// Build (hipcc is deprecated; use amdclang++ -x hip):
//   amdclang++ -x hip -ggdb -O0 --offload-arch=gfx942 \
//     $(mpicxx --showme:compile) $(mpicxx --showme:link) -o mpi_saxmy mpi_saxmy_fixed.cpp
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void saxmy(int n, float a, const float* x, float* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] * y[i];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = 256;
    const size_t bytes = n * sizeof(float);
    float *x = nullptr, *y = nullptr;
    hipMalloc(&x, bytes);
    hipMalloc(&y, bytes);        // the fix: EVERY rank allocates y

    saxmy<<<(n + 63) / 64, 64>>>(n, 2.0f, x, y);
    hipDeviceSynchronize();

    printf("rank %d done\n", rank);
    MPI_Finalize();
    return 0;
}
