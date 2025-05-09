/*
Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include <math.h>
#include "hip/hip_runtime.h"

/* Macro for checking GPU API return values */
#define gpuCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

/* --------------------------------------------------
Square elements kernel
-------------------------------------------------- */
__global__ void square_elements(float *A, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {

        A[id] = A[id] * A[id];

    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of array */
    int N = 1024 * 1024;

    /* Bytes in N ints */
    size_t bytes = N * sizeof(float);

    /* Allocate memory for host array */
    float *h_A = (float*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        h_A[i] = static_cast<float>(i);
    }

    /* Allocate memory for device array */
    float *d_A;
    gpuCheck( hipMalloc(&d_A, bytes) );

    /* Copy data from host array to device array */
    gpuCheck( hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice) );

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid */
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    /* Launch kernel */
    square_elements<<<blk_in_grid, thr_per_blk>>>(d_A, N);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device array to host array */
    gpuCheck( hipMemcpy(h_A, d_A, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    for (int i=0; i<N; i++){
        float expected = static_cast<float>(i) * static_cast<float>(i);
        if(h_A[i] != expected){
            printf("Error: h_A[%d] = %f instead of %f\n", i, h_A[i], expected );
            exit(1);
        }
    }

    /* Free CPU memory */
    free(h_A);

    /* Free Device memory */
    gpuCheck( hipFree(d_A) );

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("------------------------------\n");
    printf("N                : %d\n", N);
    printf("Blocks in Grid   : %d\n",  blk_in_grid);
    printf("Threads per Block: %d\n",  thr_per_blk);
    printf("==============================\n\n");

    return 0;
}
