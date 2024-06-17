#include <hip/hip_runtime.h>

__global__ 
void load_store(int n, float* in, float* out)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  out[tid] = in[tid];
}
