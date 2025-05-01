// Copyright AMD 2024, MIT License, contact Bob.Robey@amd.com
#include <stdio.h>
#include <hip/hip_runtime.h>

/* Macro for checking GPU API return values */
#define hipCheck(call)                                                                        \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)


__global__ void saxpy_kernel(int n, float a, float * x, float * y) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   y[i] = a * x[i] + y[i];
}


void compute_1(int n, float *x){
   for (int i=0; i<n; i++) {
      x[i] = 1.0f;  // use 1.0
   }
}

void compute_2(int n, float *y){
   for (int i=0; i<n; i++) {
      y[i] = 2.0f;  // use 2.0
   }
}

/**
 Total the results and verify the results
*/
void compute_3(int n, float *y){
   float total=0.0f;
   for (int i=0; i<n; i++) {
      total += y[i];
   }

   // expect the output to be the sum of (a * x[i] + y[i]) where
   // x[:] is initialized to 1.0, y[:] = 2.0

   if (total == (n*4.0f)) {
      printf("PASS results are verified as correct\n");
   } else {
      printf("FAIL results are not correct. Expected %f and received %f. \n", (n*4.0f), total);
   }
}


void saxpy_openmp(int n, float a, float * x, float * y);

int main(int argc, char* argv[])
{
   int n = 1024;     // use 1024 for our example
   float a = 2.0f;  // use 2.0f for our example
   float *h_x = new float[n];
   float *h_y = new float[n];
   float* d_x;
   float* d_y;

   // Size, in bytes, of each vector
   size_t bytes = n*sizeof(float);
   hipCheck( hipMalloc(&d_x, bytes) );
   hipCheck( hipMalloc(&d_y, bytes) );

   compute_1(n, h_x);
   compute_2(n, h_y);

   hipCheck( hipMemcpy(d_x, h_x, bytes, hipMemcpyHostToDevice) );
   hipCheck( hipMemcpy(d_y, h_y, bytes, hipMemcpyHostToDevice) );
   
   saxpy_openmp(n, a, d_x, d_y);  // compute a * x[i] + y[i] in parallel
      
   hipCheck( hipMemcpy(h_x, d_x, bytes, hipMemcpyDeviceToHost) );
   hipCheck( hipMemcpy(h_y, d_y, bytes, hipMemcpyDeviceToHost) );

   compute_3(n, h_y);
}
