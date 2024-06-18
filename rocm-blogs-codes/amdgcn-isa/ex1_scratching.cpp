#include <hip/hip_runtime.h>

__global__ void kernel(int* x, int len)
{
  int y[17] = {0}; //68 bytes
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < len) {
    x[i] = y[i];
  }
}

