#include "transpose_kernels.h"

#define GID(y, x, sizex) y * sizex + x
#define PAD 1

/* Use a **shared‑memory tile** (`TILE_DIM × (TILE_DIM+1)`) to stage the data.
 *    Pad the shared‑memory tile to avoid bank conflicts.
 * Load the tile from the **row‑major source** (contiguous reads).
 * `__syncthreads()`.
 * Write the transposed tile back to the **row‑major destination** (`output[col][row]`),
 *    which is now a **contiguous write** pattern.
 */

__global__ void transpose_kernel_tiled(double* __restrict input,
                                       double* __restrict output,
                                       const int rows,
                                       const int cols)
{
    // thread coordinates in the source matrix
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    // source global coordinates this thread will read
    const unsigned int srcRow = blockIdx.y * TILE_SIZE + ty;
    const unsigned int srcCol = blockIdx.x * TILE_SIZE + tx;

    // allocate a shared (LDS) memory tile with padding to avoid bank conflicts
    __shared__ double tile[TILE_SIZE][TILE_SIZE + PAD];

    // Read from global memory into tile with coalesced reads
    if (srcRow < rows && srcCol < cols) {
        tile[ty][tx] = input[GID(srcRow, srcCol, cols)];
    } else {
        tile[ty][tx] = 0.0;                // guard value – never used for writes
    }

    // Synchronize to make sure all of the tile is updated before using it
    __syncthreads();

    // destination global coordinates this thread will write
    const unsigned int dstRow = blockIdx.x * TILE_SIZE + ty; // swapped axes
    const unsigned int dstCol = blockIdx.y * TILE_SIZE + tx;

    // Write back to global memory with coalesced writes
    if (dstRow < cols && dstCol < rows) {
        output[GID(dstRow, dstCol, rows)] = tile[tx][ty];
    }
}
