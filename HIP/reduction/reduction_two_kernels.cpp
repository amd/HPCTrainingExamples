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


// Define the workgroup size (number of threads in workgroup)
// It is a multiple of 64 (wavefront size)
const static int BLOCKSIZE = 256;

// Define the grid size (number of blocks in grid)
const static int GRIDSIZE = 1024;

__global__ void get_partial_sums(const double* input, double* output, int size) {
  extern __shared__ double local_sum[];

  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;

  local_sum[threadIdx.x] = 0.0;
  for (int i = idx; i < size; i += grid_size) {
    local_sum[threadIdx.x] += input[i];
  }

  // Store local sum in shared memory
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      local_sum[threadIdx.x] += local_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = local_sum[0];
  }
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

  // Init host array
  h_in.assign(h_in.size(), 0.1f);

  // Allocate device memory
  double* d_in;
  double* d_partial_sums;
  hipCheck(hipMalloc(&d_in, N * sizeof(double)));
  hipCheck(hipMalloc(&d_partial_sums, GRIDSIZE * sizeof(double)));

  // Copy h_in into d_in
  hipCheck(hipMemcpy(d_in, h_in.data(), N * sizeof(double), hipMemcpyHostToDevice));

  // Start event timer to measure kernel timing
  hipCheck( hipEventRecord(start, NULL) );

  // Compute the partial sums
  get_partial_sums<<<GRIDSIZE, BLOCKSIZE, BLOCKSIZE*sizeof(double)>>>(d_in, d_partial_sums, N);
  get_partial_sums<<<1, GRIDSIZE, GRIDSIZE*sizeof(double)>>>(d_partial_sums, d_in, GRIDSIZE);

  // Stop event timer
  hipCheck( hipEventRecord(stop, NULL) );

  // Calculate time (in ms) for kernel
  float kernel_time;
  hipCheck( hipEventSynchronize(stop) );
  hipCheck( hipEventElapsedTime(&kernel_time, start, stop) );

  // Verify the result
  double expected_sum = 0;
  for (int i = 0; i < N; ++i) {
    expected_sum += h_in[i];
  }

  // Copy d_in[0] back to h_in, don't need the partial sums or the rest of the input array
  hipCheck(hipMemcpy(h_in.data(), d_in, 1 * sizeof(double), hipMemcpyDeviceToHost));

  std::cout << std::setprecision(14);
  //if (abs(sum - expected_sum) > 1e-7 * expected_sum) {
  //   std::cout << "FAIL: sum = " << sum <<", expected_sum = " << expected_sum << std::endl;
  if (abs(h_in[0] - expected_sum) > 1e-7 * expected_sum) {
     std::cout << "FAIL: sum = " << h_in[0] <<", expected_sum = " << expected_sum << std::endl;
  }
  else{
     std::cout<<"PASS"<<std::endl;
     std::cout<<"Kernel time: " << kernel_time << " ms" << std::endl;
  }

  hipCheck( hipFree(d_in) );
  hipCheck( hipFree(d_partial_sums) );

}

