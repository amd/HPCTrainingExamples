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

/* --------------------------------------------------
vector addition kernel
-------------------------------------------------- */
__global__ void vector_addition(double *A, double *B, double *C, int n)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    // try defining id as below, what happens? why do you think it happens?
    //int id = threadIdx.x * gridDim.x + blockIdx.x;
    if (id < n) C[id] = A[id] + B[id];
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of array */
    int N = 2560 * 2560;

    /* Bytes in array in double precision */
    size_t bytes = N * sizeof(double);

    /* Allocate memory for host arrays */
    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        h_A[i] = sin(i) * sin(i); 
        h_B[i] = cos(i) * cos(i);
        h_C[i] = 0.0;
    }    

    /* Allocate memory for device arrays */
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    /* Copy data from host arrays to device arrays */
    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice);

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid */
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    /* Launch vector addition kernel */
    vector_addition<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);

    /* Copy data from device array to host array (only need result, d_C) */
    hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);

    /* Check for correct results */
    double sum       = 0.0;
    double tolerance = 1.0e-14;

    for(int i=0; i<N; i++){
        sum = sum + h_C[i];
    } 

    if( fabs( (sum / N) - 1.0 ) > tolerance ){
        printf("Error: Sum/N = %0.2f instead of ~1.0\n", sum / N);
        exit(1);
    }

    /* Free CPU memory */
    free(h_A);
    free(h_B);
    free(h_C);

    /* Free Device memory */
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("------------------------------\n");
    printf("N                : %d\n", N);
    printf("Blocks in Grid   : %d\n", blk_in_grid);
    printf("Threads per Block: %d\n", thr_per_blk);
    printf("==============================\n\n"); 

    return 0;
}
