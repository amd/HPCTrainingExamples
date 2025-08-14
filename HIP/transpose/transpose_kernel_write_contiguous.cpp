#include "transpose_kernels.h"

/* Basic version with write contiguous memory
 Assume a 4 × 3 matrix (rows = 4, cols = 3) stored row‑major:
 After transposition we want a 3 × 4 matrix, also stored row‑major:
 cols = 3, rows = 4   cols = 4, rows = 3
 output (row‑major)  input(row_major)
 | 0  1  2  3 |       |  0  4  8 |
 | 4  5  6  7 |       |  1  5  9 |
 | 8  9 10 11 |       |  2  6 10 |
                      |  3  7 11 |
reading -- 0 4 8 1 5 9 2 6 10 3 7 11
writing -- 0 1 2 3 4 5 6 7 8 9 10 11
*/

#define GIDX(y, x, sizex) y * sizex + x

__global__ void transpose_kernel_write_contiguous(
  const double* __restrict__ input, double* __restrict__ output,
  int srcYMax, int srcXMax) {
    // Calculate destination global thread indices
    const int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    const int dstXMax = srcYMax;
    const int dstYMax = srcXMax;

    // Boundary check
    if (dstY < dstYMax && dstX < dstXMax) {
        // Transpose: output[y][x] = input[x][y]
        const int input_gid = GIDX(dstX,dstY,srcXMax); // flipped axis
        const int output_gid = GIDX(dstY,dstX,dstXMax);

        output[output_gid] = input[input_gid];
    }
}
