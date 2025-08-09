#include "transpose_kernels.h"

// Version with manual memory coalescing optimization
__global__ void transpose_lds_kernel_coalesced(float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load data in a way that maximizes coalescing
    int global_x = blockIdx.x * TILE_SIZE + tx;
    int global_y = blockIdx.y * TILE_SIZE + ty;

    if (global_x < cols && global_y < rows) {
        tile[ty][tx] = input[global_y * cols + global_x];
    } else {
        tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Transpose with optimized addressing
    int out_x = blockIdx.y * TILE_SIZE + tx;
    int out_y = blockIdx.x * TILE_SIZE + ty;

    if (out_x < rows && out_y < cols) {
        output[out_y * rows + out_x] = tile[tx][ty];
    }
}
