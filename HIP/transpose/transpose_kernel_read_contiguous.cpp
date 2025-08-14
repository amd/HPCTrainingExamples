#include "transpose_kernels.h"

/* Basic version with read contiguous memory
 Assume a 4 × 3 matrix (height = 4, width = 3) stored row‑major:
 After transposition we want a 3 × 4 matrix, also stored row‑major:
 height = 4, width = 3, height = 3  width = 4
 input (row‑major)  output(row_major)
 |  0   1   2 |       |  0  3  6  9 |
 |  3   4   5 |       |  1  4  7 10 |
 |  6   7   8 |       |  2  5  8 11 |
 |  9  10  11 |       

reading -- 0 1 2 3 4 5 6 7 8 9 10 11
writing -- 0 3 6 9 1 4 7 10 2 5 8 11
*/

#define GIDX(y, x, sizex) y * sizex + x

__global__ void transpose_kernel_read_contiguous(
  const double* __restrict__ input, double* __restrict__ output,
  int srcHeight, int srcWidth) {
    // Calculate source global thread indices
    const int srcX = blockIdx.x * blockDim.x + threadIdx.x;
    const int srcY = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (srcY < srcHeight && srcX < srcWidth) {
        // Transpose: output[x][y] = input[y][x]
        const int input_gid = GIDX(srcY,srcX,srcWidth);
        const int output_gid = GIDX(srcX,srcY,srcHeight); // flipped axis
        output[output_gid] = input[input_gid];
    }
}
