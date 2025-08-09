#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE * TILE_SIZE)

// Macro for checking GPU API return values
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

__global__ void transpose_kernel_write_contiguous(float* input, float* output, int rows, int cols);
__global__ void transpose_kernel_read_contiguous(float* input, float* output, int rows, int cols);
__global__ void transpose_lds_kernel(float* input, float* output, int rows, int cols);
__global__ void transpose_lds_kernel_optimized(float* input, float* output, int rows, int cols);
__global__ void transpose_lds_kernel_coalesced(float* input, float* output, int rows, int cols);
