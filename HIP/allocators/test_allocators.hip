#include "hip/hip_runtime.h"
#include <iostream>

#define BLOCKSIZE 64
#define SIZE 10000000

/* Macro for checking GPU API return values */
#define HIP_ASSERT(call)                                                                        \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)


__global__ void vector_add_gpu(double* a, double* b, double* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

void vector_add_cpu(double* a, double* b, double* c, int N) {
    #pragma omp parallel for
    for(int i=0; i<N; i++){
       c[i] = a[i] + b[i];
    }
}


float launch_kernel_gpu(double* a, double* b, double* c, int N) {
    hipEvent_t start, stop;
    HIP_ASSERT(hipEventCreate(&start));
    HIP_ASSERT(hipEventCreate(&stop));

    HIP_ASSERT(hipEventRecord(start));
    vector_add_gpu<<<(N + (BLOCKSIZE-1))/BLOCKSIZE, BLOCKSIZE>>>(a, b, c, N);
    HIP_ASSERT(hipEventRecord(stop));
    HIP_ASSERT(hipEventSynchronize(stop));

    float ms = 0;
    HIP_ASSERT(hipEventElapsedTime(&ms, start, stop));
    return ms;
}


float launch_kernel_cpu(double* a, double* b, double* c, int N) {
    hipEvent_t start, stop;
    HIP_ASSERT(hipEventCreate(&start));
    HIP_ASSERT(hipEventCreate(&stop));

    HIP_ASSERT(hipEventRecord(start));
    vector_add_cpu(a, b, c, N);
    HIP_ASSERT(hipEventRecord(stop));
    HIP_ASSERT(hipEventSynchronize(stop));

    float ms = 0;
    HIP_ASSERT(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

double compute_norm(double* c, int N){
    double norm=0.0;
    #pragma omp parallel for reduction(+:norm)
    for(int i=0; i<N; i++){
       norm += c[i] * c[i];
    }
    return sqrt(norm);
}

void zero_vec(double* c, int N){
    #pragma omp parallel for
    for(int i=0; i<N; i++){
       c[i] = 0.0;
    }
}

void check_for_errors(double & norm, int N){

    if(norm != 3.0 * sqrt(N)){
       std::cout<< "FAIL: vector add has not been computed correctly" << std::endl;
       abort();
    }

}


void test_allocators_cpu(double* a, double * b, double* c, int N, std::string allocator){
  float time=0.0;
  double norm=0.0;
  time = launch_kernel_cpu(a,b,c,N);
  norm=compute_norm(c,N);
  check_for_errors(norm,N);
  std::cout<<"Warm-up call - Time of vector add on CPU with "<< allocator << ": " << time << " milliseconds." <<std::endl;
  zero_vec(c,N);
  time=0.0;
  norm=0.0;
  time = launch_kernel_cpu(a,b,c,N);
  norm=compute_norm(c,N);
  check_for_errors(norm,N);
  std::cout<<"Actual call - Time of vector add on CPU with "<< allocator << ": " << time << " milliseconds." <<std::endl;
  zero_vec(c,N);
}


void test_allocators_gpu(double* a, double * b, double* c, int N, std::string allocator){
  float time=0.0;
  double norm=0.0;
  time = launch_kernel_gpu(a,b,c,N);
  norm=compute_norm(c,N);
  check_for_errors(norm,N);
  std::cout<<"Warm-up call - Time of vector add on GPU with "<< allocator << ": " << time << " milliseconds." <<std::endl;
  zero_vec(c,N);
  time=0.0;
  norm=0.0;
  time = launch_kernel_gpu(a,b,c,N);
  norm=compute_norm(c,N);
  check_for_errors(norm,N);
  std::cout<<"Actual call - Time of vector add on GPU with "<< allocator << ": " << time << " milliseconds." <<std::endl;
  zero_vec(c,N);
}


int main(){

  double* hA_malloc;
  double* hB_malloc;
  double* hC_malloc;

  double* dA_hipMalloc;
  double* dB_hipMalloc;
  double* dC_hipMalloc;

  double* hA_hipHostMalloc;
  double* hB_hipHostMalloc;
  double* hC_hipHostMalloc;

  double* dA_hipMallocManaged;
  double* dB_hipMallocManaged;
  double* dC_hipMallocManaged;

  hA_malloc = (double*)malloc(SIZE * sizeof(double));
  hB_malloc = (double*)malloc(SIZE * sizeof(double));
  hC_malloc = (double*)malloc(SIZE * sizeof(double));

  // initialize the input data
  for (int i = 0; i < SIZE; i++) {
    hA_malloc[i] = 1.0;
    hB_malloc[i] = 2.0;
    hC_malloc[i] = 0.0;
  }

  HIP_ASSERT(hipMalloc((void**)&dA_hipMalloc, SIZE * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&dB_hipMalloc, SIZE * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&dC_hipMalloc, SIZE * sizeof(double)));

  HIP_ASSERT(hipHostMalloc((void**)&hA_hipHostMalloc, SIZE * sizeof(double)));
  HIP_ASSERT(hipHostMalloc((void**)&hB_hipHostMalloc, SIZE * sizeof(double)));
  HIP_ASSERT(hipHostMalloc((void**)&hC_hipHostMalloc, SIZE * sizeof(double)));

  HIP_ASSERT(hipMallocManaged((void**)&dA_hipMallocManaged, SIZE * sizeof(double)));
  HIP_ASSERT(hipMallocManaged((void**)&dB_hipMallocManaged, SIZE * sizeof(double)));
  HIP_ASSERT(hipMallocManaged((void**)&dC_hipMallocManaged, SIZE * sizeof(double)));

  HIP_ASSERT(hipMemcpy(dA_hipMalloc, hA_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 
  HIP_ASSERT(hipMemcpy(dB_hipMalloc, hB_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 
  HIP_ASSERT(hipMemcpy(dC_hipMalloc, hC_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice));
 
  HIP_ASSERT(hipMemcpy(hA_hipHostMalloc, hA_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 
  HIP_ASSERT(hipMemcpy(hB_hipHostMalloc, hB_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 
  HIP_ASSERT(hipMemcpy(hC_hipHostMalloc, hC_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 

  HIP_ASSERT(hipMemcpy(dA_hipMallocManaged, hA_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 
  HIP_ASSERT(hipMemcpy(dB_hipMallocManaged, hB_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 
  HIP_ASSERT(hipMemcpy(dC_hipMallocManaged, hC_malloc, SIZE*sizeof(double), hipMemcpyHostToDevice)); 

  // compute on CPU - allocate with malloc
  std::string allocator = "malloc";
  test_allocators_cpu(hA_malloc,hB_malloc,hC_malloc,SIZE,allocator);

  // compute on CPU - allocate with hipMalloc
  allocator = "hipMalloc";
  test_allocators_cpu(dA_hipMalloc,dB_hipMalloc,dC_hipMalloc,SIZE,allocator);

  // compute on CPU - allocate with hipHostMalloc
  allocator = "hipHostMalloc";
  test_allocators_cpu(hA_hipHostMalloc,hB_hipHostMalloc,hC_hipHostMalloc,SIZE,allocator);

  // compute on CPU - allocate with hipMallocManaged
  allocator = "hipMallocManaged";
  test_allocators_cpu(dA_hipMallocManaged,dB_hipMallocManaged,dC_hipMallocManaged,SIZE,allocator);

  // compute on GPU - allocate with malloc
  allocator = "malloc";
  test_allocators_gpu(hA_malloc,hB_malloc,hC_malloc,SIZE,allocator);

  // compute on GPU - allocate with hipMalloc
  allocator = "hipMalloc";
  test_allocators_gpu(dA_hipMalloc,dB_hipMalloc,dC_hipMalloc,SIZE,allocator);

  // compute on GPU - allocate with hipHostMalloc
  allocator = "hipHostMalloc";
  test_allocators_gpu(hA_hipHostMalloc,hB_hipHostMalloc,hC_hipHostMalloc,SIZE,allocator);

  // compute on GPU - allocate with dC_hipMallocManaged
  allocator = "hipMallocaManaged";
  test_allocators_gpu(dA_hipMallocManaged,dB_hipMallocManaged,dC_hipMallocManaged,SIZE,allocator);

return 0;
}
