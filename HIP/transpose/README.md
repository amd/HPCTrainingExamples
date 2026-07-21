
## HIP Transpose Examples

README.md from `HPCTrainingExamples/HIP/transpose` from the Training Examples repository.

In this set of examples, we explore

* Using LDS (Local Data Share or Shared Memory)
* Coalesced reads and writes

For these exercises, retrieve them with 

```
git clone https://github.com/AMD/HPCTrainingExamples
cd HPCTrainingExamples/HIP/transpose
```

Set up your environment

```
module load rocm
```

The exercises were tested on an MI210 with ROCm version 6.4.1.

### Transpose Read Contiguous

In this example, we will read the matrix data in a contiguous 
manner. This means that the data read varies quickest by the
second index -- data[slow][fast]. This is the normal C and C++
convention. The data must be in a single block of memory. On 
the host side, we allocate the data arrays as 1D arrays. A
macro is defined on the device side to make it clearer
how the indices vary.

Examine the file `transpose_kernel_read_contiguous.cpp`. Note
that the 2D matrix is read in contiguos order and written out
with striding though memory -- Transpose: output[X][Y] = input[Y][X].

```
#define GIDX(y, x, sizex) y * sizex + x

__global__ void transpose_kernel_read_contiguous(
  double* __restrict__ input, double* __restrict__ output,
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
```

Build the transpose read contiguous application and run it.

```
make transpose_read_contiguous
./transpose_read_contiguous
```

The output for the last matrix size should look like

```
Testing Matrix dimensions: 8192 x 8192
Input size: 512.00 MB
Output size: 512.00 MB
=========================================
Basic Transpose, Read Contiguous - Average Time: 4450.20 μs
=========================================
Verification: PASSED
```

### Transpose Write Contiguous

What happens if we make the writes contiguous instead of the reads? Let's
take a look at the kernel for that case.

```
#define GIDX(y, x, sizex) y * sizex + x

__global__ void transpose_kernel_write_contiguous(
  double* __restrict__ input, double* __restrict__ output,
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
```

Now the write order for the output array is contiguous. Let's compile
and run it.

```
make transpose_write_contiguous
./transpose_write_contiguous
```

The output for the last matrix size should look like

```
Testing Matrix dimensions: 8192 x 8192
Input size: 512.00 MB
Output size: 512.00 MB
=========================================
Basic Transpose, Write Contiguous - Average Time: 2901.80 μs
=========================================
Verification: PASSED
```

We get a substantial speedup. So it is more important to have
contiguous (coalesced) writes than reads.

Can we do better than this? If we use a shared memory tile, we
can make both the read and write contiguous.

### Tiled Matrix Transpose

The kernel code for the matrix transpose with a shared memory 
tile is a little more complicated.

```
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
   double* __restrict input, double* __restrict output,
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
        output[GIDX(dstY, dstX, srcWidth)] = tile[tx][ty];
    }
}
```

Compiling and running the tiled transpose.

```
make transpose_tiled
./transpose_tiled
```

The output from the last matrix size

```
Testing Matrix dimensions: 8192 x 8192
Input size: 512.00 MB
Output size: 512.00 MB
=========================================
Tiled Transpose, Read and Write Contiguous - Average Time: 2686.40 μs
=========================================
Verification: PASSED
```

We get a little speedup over the contiguous write approach.

### Transpose from the rocblas library

Now let's try the rocblas transpose routine. We no longer
need a kernel since that will be provided by the rocblas library.
The host code is also simpler, though you do need to know how
to call the rocblas library routine.

Here is the code required to call the rocblas transpose routine

```
// See https://github.com/ROCm/rocBLAS/blob/develop/clients/samples/example_c_dgeam.c
//   for an example how to use the transpose library routine in rocblas

// Create handle to rocblas library
rocblas_handle handle;
rocblas_status roc_status=rocblas_create_handle(&handle);
CHECK_ROCBLAS_STATUS(roc_status);

// scalar arguments will be from host memory
roc_status = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
CHECK_ROCBLAS_STATUS(roc_status);

// set up the parameters needed for the transpose operation
const double alpha = 1.0;
const double beta  = 0.0;

// For transpose: C= alpha * op(A) + beta * B
// where op(A) = A^T and B is the zero matrix
rocblas_operation transa = rocblas_operation_transpose;
rocblas_operation transb = rocblas_operation_none;

// Call rocblas_geam for the transpose operation
roc_status =  rocblas_dgeam(handle,
                  transa, transb,
                  width, height,
                  &alpha, d_input, width,
                  &beta, d_output, width,
                  d_output, width);
CHECK_ROCBLAS_STATUS(roc_status);

hipCheck( hipDeviceSynchronize() );
```

Now let's build and run this version.

```
make transpose_rocblas
./transpose_rocblas
```

```
ROCBlas Transpose - Average Time: 3638.60 μs
```

So this is a little slower than some of our custom version, but it may
be because the rocblas routine has to be for general use.

#### Transpose timed comparison

For convenience, we have written a version which will run
all the transpose kernels and report a comparison between 
them.

```
make transpose_timed
./transpose_timed
```
The last part of the output should be something like:

```
Performance Summary:
Basic read contiguous   4439.60 μs
Basic write contiguous  2899.80 μs
Tiled - both contiguous 2686.80 μs
ROCBlas                 3638.60 μs
Speedup (Write Contiguous):        1.53x
Speedup (Tiled - Both Contiguous): 1.65x
Speedup (ROCBlas):                 1.22x
Verification: PASSED
```


