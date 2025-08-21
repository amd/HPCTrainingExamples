// Author: Bob Robey: Bob.Robey@amd.com

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
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

int main() {

  // Size of array to reduce
  const static int N = 128e07;

  // Create start and stop event objects for timing
  hipEvent_t start, stop;
  hipCheck( hipEventCreate(&start) );
  hipCheck( hipEventCreate(&stop) );

  // Allocate host memory
  std::vector<double> h_in(N);
  std::vector<double> h_out(1);

  // Init host array
  h_in.assign(h_in.size(), 0.1);               // fill with 0.1

  // Allocate device memory
  double *d_in = nullptr;
  double *d_out = nullptr;
  hipCheck(hipMalloc(&d_in, N * sizeof(double)));
  hipCheck(hipMalloc(&d_out, 1 * sizeof(double)));

  // Copy h_in into d_in
  hipCheck(hipMemcpy(d_in, h_in.data(), N * sizeof(double), hipMemcpyHostToDevice));

  // Start event timer to measure kernel timing
  hipCheck( hipEventRecord(start, nullptr) );

  // Reduction operation (sum)
  rocprim::plus<double> sum_op;

  // Temporary storage for rocprim::reduce
  size_t temporary_storage_size_bytes;
  void* temporary_storage_ptr = nullptr;

  // Get required size of the temporary storage
  rocprim::reduce(
     temporary_storage_ptr,
     temporary_storage_size_bytes,
     d_in,
     d_out,
     // Initial value for reduction (e.g., a large number for minimum, zero for sum)
     0.0,
     N,
     sum_op
  );

  // Allocate temporary storage
  hipCheck(hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes));

  // Perform the reduce operation
  rocprim::reduce(
     temporary_storage_ptr,
     temporary_storage_size_bytes,
     d_in,
     d_out,
     0.0,
     N,
     sum_op
  );

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
  hipCheck(hipMemcpy(h_out.data(), d_out, 1 * sizeof(double), hipMemcpyDeviceToHost));

  std::cout << std::setprecision(14);
  if (abs(h_out[0] - expected_sum) > 1e-7 * expected_sum) {
     std::cout << "FAIL: sum = " << h_out[0] <<", expected_sum = " << expected_sum << std::endl;
  }
  else{
     std::cout<<"PASS"<<std::endl;
     std::cout<<"Kernel time: " << kernel_time << " ms" << std::endl;
  }

  //  Cleanup
  hipCheck( hipFree(d_in) );
  hipCheck( hipFree(d_out) );
  hipCheck( hipFree(temporary_storage_ptr) );
  hipCheck( hipEventDestroy(start) );
  hipCheck( hipEventDestroy(stop) );
  return 0;
}
