#include <hip/hip_runtime.h>

__global__ 
void load_store(int n, float* in, float* out)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (n > tid)
    out[tid] = in[tid];
}
