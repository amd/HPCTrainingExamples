#include "transpose_kernels.h"

#define gid(y, x, sizex) y * sizex + x

// Basic version with read contiguous memory
//  cols = 3, rows = 4   cols = 4, rows = 3
//  in_gid[row][col]     out_gid[col][row]
//  |  0   1   2 |       |  0  3  6  9 |
//  |  3   4   5 |       |  1  4  7 10 |
//  |  6   7   8 |       |  2  5  8 11 |
//  |  9  10  11 |       gid (y = col, x = row, sizex = rows)
//  gid (y = row, x = col, sizex = cols)
__global__ void transpose_kernel_read_contiguous(double* input, double* output, int rows, int cols) {
    // Calculate global thread indices
    int srcCol = blockIdx.x * blockDim.x + threadIdx.x;
    int srcRow = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (srcRow < rows && srcCol < cols) {
        // Transpose: output[col][row] = input[row][col]
        // Calculate source global thread indices
        int dstRow = srcCol;
        int dstCol = srcRow;

        int input_gid = gid(srcRow,srcCol,cols);
        int output_gid = gid(dstRow,dstCol,rows);
        output[output_gid] = input[input_gid];
    }
}
