#include <math.h>
#include <hip/hip_runtime.h>

__global__ void square(double *input, int elements){
  
  size_t tid = blockIdx.x*blockDim.x+threadIdx.x;
  size_t gstride = blockDim.x * gridDim.x;
  double tmp;
  // Span all elements assigned to this stream
  for (size_t id = tid; id < elements; id += gstride){
     // Just to make the kernel take longer
     for (int i=0; i<100000; i++){
        tmp = input[id] * input[id];
     }
     input[id] = tmp;
  }
}


void hip_square(const int gridSizePerStream, const int blockSize, double * input, const int elements, hipStream_t stream) {
   square<<<gridSizePerStream, blockSize, 0, stream>>>(input, elements);
}
