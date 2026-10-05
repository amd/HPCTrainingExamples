// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// MPI + HIP saxmy: the Part II multi-rank companion to saxmy.cpp. Every rank
// runs the saxmy kernel (y = a*x*y element-wise, NOT a BLAS routine) on its own
// GPU; the LAST rank "forgets" to allocate y, so it passes a null device pointer
// and faults on the GPU - while the other ranks finish cleanly. Under the
// per-rank rocgdb wrapper that means exactly one rank's gdb_out_<rank>.txt shows
// the GPU memory fault; the rest show "done".
//
// Build (hipcc is deprecated; use amdclang++ -x hip):
//   amdclang++ -x hip -ggdb -O0 --offload-arch=gfx942 \
//     $(mpicxx --showme:compile) $(mpicxx --showme:link) -o mpi_saxmy mpi_saxmy.cpp
// Run under the per-rank wrapper (rank from SLURM_PROCID):
//   srun -n 2 --mpi=pmix ./gdb_mpi_wrapper_all_ranks.sh ./mpi_saxmy
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void saxmy(int n, float a, const float* x, float* y) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] * y[i];   // the faulting rank dereferences a null y here
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
    if (rank != size - 1)        // the last rank omits this -> null y -> GPU fault
        hipMalloc(&y, bytes);

    saxmy<<<(n + 63) / 64, 64>>>(n, 2.0f, x, y);
    hipDeviceSynchronize();      // the faulting rank reports the fault here

    printf("rank %d done\n", rank);
    MPI_Finalize();
    return 0;
}
