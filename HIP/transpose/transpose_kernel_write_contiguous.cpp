#include "transpose_kernels.h"

/* Basic version with write contiguous memory
 Assume a 3 x 4 matrix (height = 3, width = 4) stored row‑major:
 After transposition we want a 4 x 3 matrix, also stored row‑major:
 height = 3, width = 4   height = 4, width = 3
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
  int srcHeight, int srcWidth) {
    // Calculate destination global thread indices
    const int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    const int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;

    // Boundary check
    if (dstY < dstHeight && dstX < dstWidth) {
        // Transpose: output[y][x] = input[x][y]
        const int input_gid = GIDX(dstX,dstY,srcWidth); // flipped axis
        const int output_gid = GIDX(dstY,dstX,dstWidth);

        output[output_gid] = input[input_gid];
    }
}
