// Copyright AMD 2024, MIT License, contact Bob.Robey@amd.com
#include <stdio.h>
#include <assert.h>

#include <hip/hip_runtime.h>

/* Macro for checking GPU API return values */
#define hipCheck(call)                                                                        \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

__global__ void saxpy_kernel(int n, float a, float * x, float * y) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   if (i<n){
      y[i] = a * x[i] + y[i];
   }
   // for debug
#ifdef DEBUG
   printf("in kernel: y[%d]  is %g, a=%g, x[i]=%g \n", i, y[i], a, x[i]);
#endif
}

void saxpy_hip(int n, float a, float * x, float * y) {
   int threads_per_block = 256;
   int blocks_in_grid = ceil( float(n) / threads_per_block );
   saxpy_kernel<<<blocks_in_grid,threads_per_block>>>(n, a, x, y);
   hipCheck(hipDeviceSynchronize());
}
