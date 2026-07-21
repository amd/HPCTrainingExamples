#include <stdio.h>
#include <stdlib.h>

#include "hip/hip_runtime.h"

#define THREADS_PER_BLOCK_X 256

/* Macro for checking GPU API return values */

#define HIP_CHECK(call)                                   \
do{                                                      \
    hipError_t gpuErr = call;                            \
    if(hipSuccess != gpuErr){                            \
        printf("GPU API Error - %s:%d: '%s'\n",          \
         __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                         \
    }                                                    \
}while(0)

__global__ void
gpu_func(double *in_h, double *out_h, int M){
   const int id = blockDim.x * blockIdx.x + threadIdx.x;
   if ( id < M) {
      out_h[id] = in_h[id] * 2.0;
   }
}

int main(int argc, char *argv[]) {
   int M=100000;
   size_t Msize=M*sizeof(double);
   double sum=0.0;

   double *in_h = (double*)malloc(Msize);
   double *out_h = (double*)malloc(Msize);

   for (int i=0; i<M; i++) // initialize
      in_h[i] = 1.0;

   dim3 grid((M+THREADS_PER_BLOCK_X-1)/THREADS_PER_BLOCK_X,1,1);
   dim3 block(THREADS_PER_BLOCK_X,1,1);
   gpu_func<<<grid,block,0,0>>>(in_h, out_h, M);

   HIP_CHECK(hipDeviceSynchronize());

   for (int i=0; i<M; i++) // CPU-process
     sum += out_h[i];

   printf("Result is %lf\n",sum);

   free(in_h);
   free(out_h);
}
