// daxpy_kernel.hip.cpp
#include <hip/hip_runtime.h>
#include <stdio.h>


extern "C" {

__global__ void daxpy_kernel(int n, double a, double * x, double * y) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   y[i] = a * x[i] + y[i];
   // for debug
#ifdef DEBUG
   printf("in kernel: y[%d]  is %g, a=%g, x[i]=%g \n", i, y[i], a, x[i]);
#endif
}

void daxpy_hip(int n, double a, double * x, double * y) {
   printf("daxpy_hip Compiled with DEVICE_CODE \n");
   assert(n % 256 == 0);
   daxpy_kernel<<<n/256,256,0,NULL>>>(n, a, x, y);
   int ret=hipDeviceSynchronize();
}

}
