// Original code by Yifan Sun: https://gitlab.com/syifan/hipbookexample
// Modified by Giacomo Capodaglio: Giacomo.Capodaglio@amd.com

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

__global__ void get_partial_sums_to_fix(const double* input, double* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_size = gridDim.x * blockDim.x;

  double local_sum = 0.0;
  for (int i = idx; i < size; i += global_size) {
    local_sum += input[i];
  }

  output[blockIdx.x] = local_sum;
}

int main() {

  // Size of array to reduce
  const static int N = 128e07;

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
  double* d_partial_sums;
  hipCheck( hipMalloc(&d_in, N * sizeof(double)) );
  hipCheck( hipMalloc(&d_partial_sums, GRIDSIZE * sizeof(double)) );

  // Copy h_in into d_in
  hipCheck( hipMemcpy(d_in, h_in.data(), N * sizeof(double), hipMemcpyHostToDevice) );

  // Start event timer to measure kernel timing
  hipCheck( hipEventRecord(start, NULL) );

  // Compute the partial sums
  get_partial_sums_to_fix<<<GRIDSIZE, BLOCKSIZE>>>(d_in, d_partial_sums, N);

  // Stop event timer
  hipCheck( hipEventRecord(stop, NULL) );

  // Calculate time (in ms) for kernel
  float kernel_time;
  hipCheck( hipEventSynchronize(stop) );
  hipCheck( hipEventElapsedTime(&kernel_time, start, stop) );

  // Copy d_in back to h_in
  hipCheck( hipMemcpy(h_partial_sums.data(), d_partial_sums, GRIDSIZE * sizeof(double), hipMemcpyDeviceToHost) );

  // Compute the actual reduction from the partial sum
  double sum = 0.0;
  for (int i = 0; i < GRIDSIZE; ++i) {
     sum += h_partial_sums[i];
  }

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
  hipCheck( hipFree(d_partial_sums) );

}

