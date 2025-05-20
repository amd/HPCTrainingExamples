#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

/* Macro for checking GPU API return values */
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

// Defined the workgroup size (number of threads in workgroup)
// It is a multiple of 64 (wavefront size)
const static int BLOCKSIZE = 1024;

// Define the grid size (number of blocks in grid)
const static int GRIDSIZE = 1024;

__global__ void atomic_reduction(const double* input, double* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_size = gridDim.x * blockDim.x;

  double local_sum = 0.0;
  for (int i = idx; i < size; i += global_size) {
    local_sum += input[i];
  }

  //unsafeAtomicAdd(output,local_sum);
  atomicAdd(output,local_sum);
}

int main() {

  // Size of array to reduce
  const static int N = 128e07;

  // Defined the workgroup size (number of threads in workgroup)
  // It is a multiple of 64 (wavefront size)
  const static int BLOCKSIZE = 256;

  // Create start and stop event objects for timing
  hipEvent_t start, stop;
  hipCheck( hipEventCreate(&start) );
  hipCheck( hipEventCreate(&stop) );

  // Allocate host memory
  std::vector<double> h_in(N);
  std::vector<double> h_partial_sums(GRIDSIZE);

  // Initialize host array
  h_in.assign(h_in.size(), 0.1);

  // Allocate device memory
  double* d_in;
  double* d_out;
  hipCheck( hipMalloc(&d_in, N * sizeof(double)) );
  hipCheck( hipMalloc(&d_out, sizeof(double)) );

  // Copy h_in into d_in
  hipCheck( hipMemcpy(d_in, h_in.data(), N * sizeof(double), hipMemcpyHostToDevice) );

  // Start event timer to measure kernel timing
  hipCheck( hipEventRecord(start, NULL) );

  // Compute the partial sums
  atomic_reduction<<<GRIDSIZE, BLOCKSIZE>>>(d_in, d_out, N);

  // Stop event timer
  hipCheck( hipEventRecord(stop, NULL) );

  // Calculate time (in ms) for kernel
  float kernel_time;
  hipCheck( hipEventSynchronize(stop) );
  hipCheck( hipEventElapsedTime(&kernel_time, start, stop) );

  // Copy the device sum to the host
  double sum;
  hipCheck( hipMemcpy(&sum, d_out, sizeof(double), hipMemcpyDeviceToHost) );

  // Verify the result.
  double expected_sum = 0.0;
  for (int i = 0; i < N; ++i) {
    expected_sum += h_in[i];
  }

  std::cout << std::setprecision(14);
  if (abs(sum - expected_sum) > 1e-7 * expected_sum) {
     std::cout << "FAIL: sum = " << sum <<", expected_sum = " << expected_sum << std::endl;
  }
  else{
     std::cout<<"PASS"<<std::endl;
     std::cout<<"Kernel time: " << kernel_time << " ms" << std::endl;     
  }

  hipCheck( hipFree(d_in) );
  hipCheck( hipFree(d_out) );

}

