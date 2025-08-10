#include "transpose_kernels.h"

/* Basic version with write contiguous memory
 Assume a 4 × 3 matrix (rows = 4, cols = 3) stored row‑major:

input (row‑major)          memory layout (linear indices)
0  1  2                     0 1 2
3  4  5                     3 4 5
6  7  8                     6 7 8
9 10 11                     9 10 11

After transposition we want a 3 × 4 matrix, also stored row‑major:

output (row‑major)         memory layout (linear indices)
0  3  6  9                  0 1 2 3 4 5 6 7 8 9 10 11
1  4  7 10
2  5  8 11
*/

#define GID(y, x, sizex) y * sizex + x

__global__
void transpose_kernel_write_contiguous(double* __restrict__ input,
                                       double* __restrict__ output,
                                       int rows, int cols) {

    // Calculate destination global thread indices
    int dstRow = blockIdx.y * blockDim.y + threadIdx.y;
    int dstCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (dstRow < cols && dstCol < rows) {
        // Calculate source global thread indices
        int srcRow = dstCol;
        int srcCol = dstRow;

        // Transpose: output[row][col] = input[col][row]
        int input_gid = GID(srcRow,srcCol,cols);
        int output_gid = GID(dstRow,dstCol,rows);

        output[output_gid] = input[input_gid];
    }
}
