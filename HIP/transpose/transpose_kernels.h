#ifndef TRANSPOSE_KERNELS_H
#define TRANSPOSE_KERNELS_H

#include <hip/hip_runtime.h>

#define TILE_SIZE 32
#define BLOCK_SIZE (TILE_SIZE * TILE_SIZE)
#define PAD 1                 // padding to avoid bank conflicts

__global__
void transpose_kernel_read_contiguous(const double* __restrict input,
                                      double* __restrict output,
                                      const int height,
                                      const int width);

__global__
void transpose_kernel_write_contiguous(const double* __restrict input,
                                       double* __restrict output,
                                       const int height,
                                       const int width);

__global__
void transpose_kernel_tiled(const double* __restrict input,
                            double* __restrict output,
                            const int height,
                            const int width);

#endif // TRANSPOSE_KERNELS_H
