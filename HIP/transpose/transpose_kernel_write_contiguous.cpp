#include "transpose_kernels.h"

// Basic version with LDS transpose
__global__ void transpose_kernel_write_contiguous(float* input, float* output, int rows, int cols) {
    // Calculate global thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (row < rows && col < cols) {
        // Transpose: output[row][col] = input[col][row]
        int input_idx = col * rows + row;
        int output_idx = row * cols + col;
        output[output_idx] = input[input_idx];
    }
}
