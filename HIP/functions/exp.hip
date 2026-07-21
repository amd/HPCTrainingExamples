#include <hip/hip_runtime.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <omp.h>

// Macro for checking GPU API return values
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

__host__ __device__ double compute_term(int index, double value){

   double denom = index;
   double num = value;
   for(int i=1; i<index; i++){
     denom *= (double)i;
     num *= value;
   }
   return num/denom;

}

__global__ void gpu_get_Taylor_terms_stride(int num_terms, double* d_terms, double value){

   int id = blockIdx.x * blockDim.x + threadIdx.x;
   int stride_size =  blockDim.x * gridDim.x;
   for(int i=id; i < num_terms; i+=stride_size){
      d_terms[i] = compute_term(num_terms-i,value);
   }

}

__global__ void gpu_get_Taylor_terms(int num_terms, double* d_terms, double value){

   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if( id < num_terms) d_terms[id] = compute_term(num_terms-id,value);

}

void cpu_get_Taylor_terms(int num_terms, double* &h_terms, double value){
#pragma omp parallel for
   for(int i=1; i<num_terms;i++){
      h_terms[i] = compute_term(num_terms-i,value);
   }
}

int main(int argc, char* argv[]) {

int num_terms = 100;
double x = 10;

double* d_terms;
double* h_terms;

hipCheck(hipMalloc((void**)&d_terms, num_terms * sizeof(double)));
h_terms = (double*)malloc(num_terms * sizeof(double));

//Creating events for timers
hipEvent_t start, stop;
hipCheck(hipEventCreate(&start));
hipCheck(hipEventCreate(&stop));

double expApprox = 1.0;

hipCheck(hipEventRecord(start));
// parallel on the CPU
cpu_get_Taylor_terms(num_terms, h_terms, x);
hipCheck(hipEventRecord(stop));
hipCheck(hipEventSynchronize(stop));
float milliseconds = 0;
hipCheck(hipEventElapsedTime(&milliseconds, start, stop));
std::cout<<"Time for the computation on the CPU in ms: "<< milliseconds << std::endl;

int threads_per_block = 64;
int blocks_in_grid = (float(num_terms) + threads_per_block -1 ) / threads_per_block ;
hipCheck(hipEventRecord(start));
// parallel on the GPU
//gpu_get_Taylor_terms<<<threads_per_block,blocks_in_grid>>>(num_terms,d_terms,x);
gpu_get_Taylor_terms_stride<<<threads_per_block,blocks_in_grid>>>(num_terms,d_terms,x);
hipCheck(hipEventRecord(stop));
hipCheck(hipEventSynchronize(stop));
milliseconds = 0;
hipCheck(hipEventElapsedTime(&milliseconds, start, stop));
std::cout<<"Time for the computation on the GPU in ms: "<< milliseconds << std::endl;

// compute reduction for the host first
#pragma omp parallel for reduction (+:expApprox)
for(int i=0; i<num_terms;i++){
   expApprox += h_terms[i];
}

if(fabs(expApprox-exp(x)) < 1.e-10){
   std::cout<<"PASS ON CPU"<<std::endl;
}
else {
   std::cout<<"FAIL ON CPU"<<std::endl;
   std::setprecision(16);
   std::cout<<"Expected: " << std::setprecision(16) << exp(x) << " Got: " << std::setprecision(16) << expApprox << std::endl;
}


// then compute the reduction for the device by overwriting h_terms
hipCheck(hipMemcpy(h_terms, d_terms, num_terms * sizeof(double), hipMemcpyDeviceToHost));

expApprox=1.0;
#pragma omp parallel for reduction (+:expApprox)
for(int i=0; i<num_terms;i++){
   expApprox += h_terms[i];
}

if(fabs(expApprox-exp(x)) < 1.e-10){
   std::cout<<"PASS ON GPU"<<std::endl;
}
else {
   std::cout<<"FAIL ON GPU"<<std::endl;
   std::cout<<"Expected: " << std::setprecision(16) << exp(x) << " Got: " << std::setprecision(16) << expApprox << std::endl;
}


}
