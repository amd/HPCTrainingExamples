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
Matrix addition kernel
-------------------------------------------------- */
__global__ void matrix_addition(double *A, double *B, double *C, int n)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < n && row < n){
        int index = n * row + col;
        C[index] = A[index] + B[index];
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Size of NxN matrix */
    int N = 1024;

    /* Bytes in matrix in double precision */
    size_t bytes = N * N * sizeof(double);

    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    /* Initialize host arrays */
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            int index = N * i + j;
            h_A[index] = sin(index) * sin(index); 
            h_B[index] = cos(index) * cos(index);
            h_C[index] = 0.0;
        }
    }    

    /* Allocate memory for device matrices */
    double *d_A, *d_B, *d_C;
    gpuCheck( hipMalloc(&d_A, bytes) );
    gpuCheck( hipMalloc(&d_B, bytes) );
    gpuCheck( hipMalloc(&d_C, bytes) );

    /* Copy data from host matrices to device matricesL */
    gpuCheck( hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice) );
    gpuCheck( hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice) );

    /* Set kernel configuration parameters
           thr_per_blk: number of threads per thread block
           blk_in_grid: number of thread blocks in grid
    
       (NOTE: dim3 is a c struct with member variables x, y, z) */
    dim3 thr_per_blk( 16, 16, 1 );
    dim3 blk_in_grid( ceil( float(N) / thr_per_blk.x), ceil(float(N) / thr_per_blk.y), 1 );

    /* Launch matrix addition kernel */
    matrix_addition<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    gpuCheck( hipGetLastError() );

    /* Check for kernel execution errors */
    gpuCheck ( hipDeviceSynchronize() );

    /* Copy data from device matrix to host matrix (only need result, d_C) */
    gpuCheck( hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost) );

    /* Check for correct results */
    double sum       = 0.0;
    double tolerance = 1.0e-14;

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            int index = N * i + j;
            sum = sum + h_C[index];
        }
    } 

    if( fabs( (sum / (N*N)) - 1.0 ) > tolerance ){
        printf("Error: Sum/(N*N) = %0.14f instead of 1.00000000000000\n", sum / (N*N));
        exit(1);
    }

    /* Free CPU memory */
    free(h_A);
    free(h_B);
    free(h_C);

    /* Free Device memory */
    gpuCheck( hipFree(d_A) );
    gpuCheck( hipFree(d_B) );
    gpuCheck( hipFree(d_C) );

    printf("\n==============================\n");
    printf("__SUCCESS__\n");
    printf("------------------------------\n");
    printf("N                  : %d\n", N);
    printf("X Blocks in Grid   : %d\n",  blk_in_grid.x);
    printf("X Threads per Block: %d\n",  thr_per_blk.x);
    printf("Y Blocks in Grid   : %d\n",  blk_in_grid.y);
    printf("Y Threads per Block: %d\n",  thr_per_blk.y);
    printf("==============================\n\n");

    return 0;
}
