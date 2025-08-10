#ifndef TRANSPOSE_KERNELS_H
#define TRANSPOSE_KERNELS_H

#include <hip/hip_runtime.h>

#define TILE_SIZE 32
#define BLOCK_SIZE (TILE_SIZE * TILE_SIZE)
#define PAD 1                 // padding to avoid bank conflicts

__global__
void transpose_kernel_read_contiguous(double* __restrict input,
                                      double* __restrict output,
                                      const int rows,
                                      const int cols);

__global__
void transpose_kernel_write_contiguous(double* __restrict input,
                                       double* __restrict output,
                                       const int rows,
                                       const int cols);

__global__
void transpose_kernel_tiled(double* __restrict input,
                            double* __restrict output,
                            const int rows,
                            const int cols);

#endif // TRANSPOSE_KERNELS_H
