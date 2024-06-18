#include <hip/hip_runtime.h>

__global__ void shifted_copy (float *in, float *out) {
  size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  out[gid] = in[gid+4];
}
