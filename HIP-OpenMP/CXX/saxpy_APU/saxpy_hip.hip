#include <stdio.h>

// N must be a multiple of 256
#include <hip/hip_runtime.h>

__global__ void saxpy_kernel(int n, float a, float * x, float * y) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   y[i] = a * x[i] + y[i];
   // for debug
#ifdef DEBUG
   printf("in kernel: y[%d]  is %g, a=%g, x[i]=%g \n", i, y[i], a, x[i]);
#endif
}

void saxpy_hip(int n, float a, float * x, float * y) {
   assert(n % 256 == 0);
   saxpy_kernel<<<n/256,256,0,NULL>>>(n, a, x, y);
   int ret=hipDeviceSynchronize();
}
