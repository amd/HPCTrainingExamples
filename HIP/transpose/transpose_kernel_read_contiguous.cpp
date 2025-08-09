#include "transpose_kernels.h"

// Basic version with LDS transpose
__global__ void transpose_kernel_read_contiguous(float* input, float* output, int rows, int cols) {
    // Calculate global thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (row < rows && col < cols) {
        // Transpose: output[col][row] = input[row][col]
        int input_idx = row * cols + col;
        int output_idx = col * rows + row;
        output[output_idx] = input[input_idx];
    }
}
