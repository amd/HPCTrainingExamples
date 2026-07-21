#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include"hipCheck.h"


/* --------------------------------------------------
Matrix multiply kernel
-------------------------------------------------- */
extern "C" {
   __global__ void matrix_multiply_kernel(double *A, double *B, double *C, int n)
   {
       int col = blockDim.x * blockIdx.x + threadIdx.x;
       int row = blockDim.y * blockIdx.y + threadIdx.y;

       if (col < n && row < n){

           int index = row * n + col;
           double element = 0.0;

           for (int i=0; i<n; i++){

               int row_index = n * row + i;
               int col_index = n * i   + col;

               element = element + A[col_index] * B[row_index]; 
           }

           C[index] = element;
       }
   }

   void matrix_multiply(double* A, double* B, double* C, int &n){

    // important note: the 1D arrays A and B are the transpose
    // of the corresponding arrays in Fortran: for example
    // A[1] here corresponds to A(2)(1) in Fortran.
    // Fortran has column-major ordering. 

    // define thread grid
    dim3 thr_per_blk( 16, 16, 1 );
    dim3 blk_in_grid( ceil( float(n) / thr_per_blk.x), ceil(float(n) / thr_per_blk.y), 1 );

    // declare device arrays
    double* dA;
    double* dB;
    double* dC;

    // allocation on device
    hipCheck(hipMalloc(&dA, n*n*sizeof(double)));
    hipCheck(hipMalloc(&dB, n*n*sizeof(double)));
    hipCheck(hipMalloc(&dC, n*n*sizeof(double)));

    // copy from host to device
    hipCheck(hipMemcpy(dA, A, n*n*sizeof(double), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(dB, B, n*n*sizeof(double), hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(dC, C, n*n*sizeof(double), hipMemcpyHostToDevice));

    // kernel launch
    matrix_multiply_kernel<<<blk_in_grid, thr_per_blk>>>(dA, dB, dC, n);

    // copy back from device to host
    hipCheck(hipMemcpy(A, dA, n*n*sizeof(double), hipMemcpyDeviceToHost));
    hipCheck(hipMemcpy(B, dB, n*n*sizeof(double), hipMemcpyDeviceToHost));
    hipCheck(hipMemcpy(C, dC, n*n*sizeof(double), hipMemcpyDeviceToHost));

    // important note: the matrix C will be automatically transposed 
    // when it is given back to Fortran:
    // for example the first n entries of C here will make up
    // the first column of C in Fortran.
    // this detail is taken into consideration in the matrix_multiply kernel
    // when defining the variable called index 
    // (by row-major so the transpose will be column-major)

    // free memory
    hipCheck(hipFree(dA));
    hipCheck(hipFree(dB));
    hipCheck(hipFree(dC));

   }
}
