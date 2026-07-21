#include "transpose_kernels.h"

#define GIDX(y, x, sizex) y * sizex + x
#define PAD 1

/* Use a **shared‑memory tile** (`TILE_SIZE × (TILE_SIZE+PAD)`) to stage the data.
 *    Pad the shared‑memory tile to avoid bank conflicts.
 * Load the tile from the **row‑major source** (contiguous reads).
 * `__syncthreads()`.
 * Write the transposed tile back to the **row‑major destination** (`output[col][row]`),
 *    which is now a **contiguous write** pattern.
 */

__global__ void transpose_kernel_tiled(
   const double* __restrict input, double* __restrict output,
   const int srcHeight, const int srcWidth)
{
    // thread coordinates in the source matrix
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // source global coordinates this thread will read
    const int srcX = blockIdx.x * TILE_SIZE + tx;
    const int srcY = blockIdx.y * TILE_SIZE + ty;

    // allocate a shared (LDS) memory tile with padding to avoid bank conflicts
    __shared__ double tile[TILE_SIZE][TILE_SIZE + PAD];

    // Read from global memory into tile with coalesced reads
    if (srcY < srcHeight && srcX < srcWidth) {
        tile[ty][tx] = input[GIDX(srcY, srcX, srcWidth)];
    } else {
        tile[ty][tx] = 0.0;                // guard value – never used for writes
    }

    // Synchronize to make sure all of the tile is updated before using it
    __syncthreads();

    // destination global coordinates this thread will write
    const int dstY = blockIdx.x * TILE_SIZE + ty; // swapped axes
    const int dstX = blockIdx.y * TILE_SIZE + tx;

    // Write back to global memory with coalesced writes
    if (dstY < srcWidth && dstX < srcHeight) {
        output[GIDX(dstY, dstX, srcHeight)] = tile[tx][ty];
    }
}
