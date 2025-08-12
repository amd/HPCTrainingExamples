# Transpose Examples

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

## Transpose Read Contiguous

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
  int srcYMax, int srcXMax) {
    // Calculate source global thread indices
    const int srcX = blockIdx.x * blockDim.x + threadIdx.x;
    const int srcY = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (srcY < srcYMax && srcX < srcXMax) {
        // Transpose: output[x][y] = input[y][x]
        const int input_gid = GIDX(srcY,srcX,srcXMax);
        const int output_gid = GIDX(srcX,srcY,srcYMax); // flipped axis
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


