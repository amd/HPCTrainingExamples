// Author: Bob Robey: Bob.Robey@amd.com

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
constexpr int BLOCKSIZE = 1024;

// Define the grid size (number of blocks in grid)
constexpr int GRIDSIZE = 1024;

// ---------------------------------------------------------------------
//  reduction_to_array
//    * each thread accumulates a strided sum over the input array
//    * the per thread sum is reduced inside its own wavefront with
//      __shfl_down
//    * the wavefront leader (lane 0) writes its partial sum to a tiny
//      shared memory array (one double per wavefront)
//    * the first wavefront reduces those per wavefront sums to the final
//      block sum and writes it to output[blockIdx.x]
// ---------------------------------------------------------------------
__global__ void reduction_to_array(const double* __restrict__ input,
                                   double*       __restrict__ output,
                                   int size)
{
  // Global ID of thread in thread grid
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Stride size is equal to total number of threads in grid
  int grid_size = blockDim.x * gridDim.x;
  int tid   = threadIdx.x;

  double threadSum = 0.0;
  for (int i = idx; i < size; i += grid_size) {
      threadSum += input[i];
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      threadSum += __shfl_down(threadSum, offset);
  }

  //  Write each wavefront's leader (lane 0) to shared memory
  const int lane   = tid % warpSize;               // 0 … 63
  const int warpId = tid / warpSize;               // 0 … (BLOCKSIZE/warpSize-1)

  // one double per wavefront – the kernel launch supplies the exact size
  extern __shared__ double warpSums[];
  if (lane == 0) {
      warpSums[warpId] = threadSum;                // one value per wavefront
  }

  __syncthreads();                                 // make sure all warp sums are visible

  //  The first wavefront reduces the per wavefront sums.
  double blockSum = 0.0;
  if (warpId == 0) {                               // only the first wavefront participates
      // load the warp sums into the lanes of the first wavefront
      if (lane < (blockDim.x / warpSize)) {
          blockSum = warpSums[lane];
      }

      // final reduction inside the first wavefront
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
          blockSum += __shfl_down(blockSum, offset);
      }

      // lane 0 now holds the block wide sum
      if (lane == 0) {
          output[blockIdx.x] = blockSum;
      }
  }
}

int main() {

  // Size of array to reduce
  const static int N = 128e07;

  // Create start and stop event objects for timing
  hipEvent_t start, stop;
  hipCheck( hipEventCreate(&start) );
  hipCheck( hipEventCreate(&stop) );

  if( GRIDSIZE % warpSize != 0){
     std::cout<<"ERROR: GRIDSIZE needs to be a multiple of " << warpSize << " in this example" << std::endl;
     abort();
  }

  if( BLOCKSIZE % warpSize != 0){
     std::cout<<"ERROR: BLOCKSIZE needs to be a multiple of " << warpSize << " in this example" << std::endl;
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

  // Start event timer to measure kernel timing
  hipCheck( hipEventRecord(start, nullptr) );

  // Compute the reductions
  reduction_to_array<<<GRIDSIZE, BLOCKSIZE, (BLOCKSIZE / warpSize) * sizeof(double)>>>(d_in, d_partial_sums, N);
  reduction_to_array<<<1, GRIDSIZE, (GRIDSIZE / warpSize) * sizeof(double)>>>(d_partial_sums, d_in, GRIDSIZE);

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
