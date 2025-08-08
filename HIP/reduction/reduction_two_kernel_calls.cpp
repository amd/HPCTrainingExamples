// Original code by Yifan Sun: https://gitlab.com/syifan/hipbookexample
// Modified by Giacomo Capodaglio: Giacomo.Capodaglio@amd.com
// and Bob Robey: Bob.Robey@amd.com
// see also "Programming in Parallel with CUDA, a Practical Guide by Richard Ansorge"

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
const static int BLOCKSIZE = 1024;

// Define the grid size (number of blocks in grid)
const static int GRIDSIZE = 1024;

__global__ void reduction_to_array(const double* input,
                                   double* output,
                                   int size)
{
  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;

  extern __shared__ double local_sum[];
  local_sum[threadIdx.x] = 0.0;
  for (int i = idx; i < size; i += grid_size) {
     local_sum[threadIdx.x] += input[i];
  }

  __syncthreads();

  // note that in this for loop we are using blockDim.x
  // since this kernel will be called twice with
  // two different grids
  // we are also assuming blockDim.x  is a power of 2
  // and that it is not larger than 1024
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

  if( (GRIDSIZE & (GRIDSIZE - 1)) != 0){
     std::cout<<"ERROR: GRIDSIZE needs to be a power of 2 in this example" << std::endl;
     abort();
  }

  if( (BLOCKSIZE & (BLOCKSIZE - 1)) != 0){
     std::cout<<"ERROR: BLOCKSIZE needs to be a power of 2 in this example" << std::endl;
     abort();
  }

  if ( GRIDSIZE > 1024 ) {
     std::cout<<"ERROR: GRIDSIZE cannot be larger than 1024, see README for reason why" << std::endl;
     abort();
  }

  // Allocate host memory
  std::vector<double> h_in(N);

  // Init host array
  h_in.assign(h_in.size(), 0.1);               // fill with 0.1

  // Allocate device memory
  double *d_in = nullptr;
  double *d_partial_sums = nullptr;
  hipCheck(hipMalloc(&d_in, N * sizeof(double)));
  hipCheck(hipMalloc(&d_partial_sums, GRIDSIZE * sizeof(double)));

  // Copy h_in into d_in
  hipCheck(hipMemcpy(d_in, h_in.data(), N * sizeof(double), hipMemcpyHostToDevice));

  //  Shared‑memory size = (BLOCKSIZE / warpSize) * sizeof(double)
  const size_t shmem_per_block = BLOCKSIZE  * sizeof(double);
  const size_t shmem_final = GRIDSIZE * sizeof(double);

  // Start event timer to measure kernel timing
  hipCheck( hipEventRecord(start, nullptr) );

  // Compute the reductions
  reduction_to_array<<<GRIDSIZE, BLOCKSIZE, shmem_per_block>>>(d_in, d_partial_sums, N);
  //  Second launch: one block, GRIDSIZE threads
  reduction_to_array<<<1, GRIDSIZE, shmem_final>>>(d_partial_sums, d_in, GRIDSIZE);

  // Stop event timer
  hipCheck( hipEventRecord(stop, nullptr) );

  // Calculate time (in ms) for kernel
  float kernel_time = 0.0f;
  hipCheck( hipEventSynchronize(stop) );
  hipCheck( hipEventElapsedTime(&kernel_time, start, stop) );

  // Verify the result
  double expected_sum = 0.0;
  for (int i = 0; i < N; ++i) {
    expected_sum += h_in[i];
  }

  // Copy d_in[0] back to h_in, don't need the partial sums or the rest of the input array
  hipCheck(hipMemcpy(h_in.data(), d_in, 1 * sizeof(double), hipMemcpyDeviceToHost));

  std::cout << std::setprecision(14);
  if (abs(h_in[0] - expected_sum) > 1e-7 * expected_sum) {
     std::cout << "FAIL: sum = " << h_in[0] <<", expected_sum = " << expected_sum << std::endl;
  }
  else{
     std::cout<<"PASS"<<std::endl;
     std::cout<<"Kernel time: " << kernel_time << " ms" << std::endl;
  }

  //  Cleanup
  hipCheck( hipFree(d_in) );
  hipCheck( hipFree(d_partial_sums) );
  hipCheck( hipEventDestroy(start) );
  hipCheck( hipEventDestroy(stop) );
  return 0;
}
