/*
   https://github.com/LLNL/RAJAPerf/blob/12b07bc19835cef3931232ec931ccbfa62202ad5/src/basic/NESTED_INIT-Hip.cpp
   See exp10 and exp11.
*/
#include <iostream>
#include <hip/hip_runtime.h>
#include "Timer.hpp"

#define BLOCKDIMX 256
#define BLOCKDIMY 1
#define BLOCKDIMZ 1

#define HIP_CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

#define NESTED_INIT_BODY  \
  array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;

using Real = float;
using Index_type = int;

// --- slow
__global__ void nested_init_exp10(Real* array,
                            Index_type ni, Index_type nj, Index_type nk, Index_type imin)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  Index_type j = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type k = blockIdx.z;

  if (  i > imin && i < ni && j < nj && k < nk ) {
    array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;
  }
}  

// --- the fix
__global__ void nested_init_exp11(Real* array,
                            Index_type ni, Index_type nj, Index_type nk,
                            Index_type i_offset, Index_type j_offset, Index_type k_offset)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  Index_type j = blockIdx.y * blockDim.y + threadIdx.y;
  Index_type k = blockIdx.z;

  if (  (i+i_offset) < (ni+i_offset) && (j+j_offset) < (nj+j_offset) && (k+k_offset) < (nk+k_offset) ) {
    array[(i+i_offset)+(ni+i_offset)*((j+j_offset)+nj*(k+k_offset))] = 0.00000001 * (i+i_offset) * (j+j_offset) * (k+k_offset) ;
  }
}

int main(int argc, char** argv)
{
  size_t n = 1<<24;
  std::cout << "n: " << n << "\n";
  size_t nbytes = n * sizeof(Real);
  size_t nwarmups = 10;
  size_t niters = 100;
  Timer time;
  float gb, tavg, bw;

  Real *array;
  hipMalloc(&array, nbytes);

  size_t nthreads = BLOCKDIMX;
  size_t nblocks = (n-1)/nthreads + 1;
  dim3 block(nthreads, 1, 1);
  dim3 grid(nblocks, 1, 1);

  int ni = n;
  int nj = 1;
  int nk = 1;
  int imin = 0;

  // Warmups
  for (size_t i = 0; i < nwarmups; ++i)
    nested_init_exp10<<<grid, block>>>(array, ni, nj, nk, imin);

  // Performance runs 1
  time.start();
  for (size_t i = 0; i < niters; ++i)
  {
    nested_init_exp10<<<grid, block>>>(array, ni, nj, nk, imin);
    HIP_CHECK(hipGetLastError());
  }
  time.stop();
  // BW
  gb = nbytes * 1.e-9;
  tavg = time.elapsedms()/niters;
  bw = gb / tavg * 1.e3;
  std::cout << "HBM (GB): " << gb
            << ", time_avg (ms): " << tavg
            << ", HBM BW (GB/s): " << bw << "\n";

  // Performance runs 2
  int i_off = 0;
  int j_off = 0;
  int k_off = 0;
  time.start();
  for (size_t i = 0; i < niters; ++i)
  {
    nested_init_exp11<<<grid, block>>>(array, ni, nj, nk, i_off, j_off, k_off);
    HIP_CHECK(hipGetLastError());
  }
  time.stop();
  // BW
  gb = nbytes * 1.e-9;
  tavg = time.elapsedms()/niters;
  bw = gb / tavg * 1.e3;
  std::cout << "HBM (GB): " << gb
            << ", time_avg (ms): " << tavg
            << ", HBM BW (GB/s): " << bw << "\n";

  hipFree(array);

  return EXIT_SUCCESS;
}
