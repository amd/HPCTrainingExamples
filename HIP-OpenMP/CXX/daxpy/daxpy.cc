// Copyright AMD 2024, MIT License, contact Bob.Robey@amd.com
#include <stdio.h>

#ifdef __DEVICE_CODE__
#include <hip/hip_runtime.h>

__global__ void daxpy_kernel(int n, double a, double * x, double * y) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   y[i] = a * x[i] + y[i];
   // for debug
#ifdef DEBUG
   printf("in kernel: y[%d]  is %g, a=%g, x[i]=%g \n", i, y[i], a, x[i]);
#endif
}

void daxpy_hip(int n, double a, double * x, double * y) {
   printf("daxpy_hip Compiled with DEVICE_CODE \n");
   assert(n % 256 == 0);
   daxpy_kernel<<<n/256,256,0,NULL>>>(n, a, x, y);
   int ret=hipDeviceSynchronize();
}
#endif


#ifdef __HOST_CODE__

void compute_1(int n, double *x){
   for (int i=0; i<n; i++) {
      x[i] = 1.0;  // use 1.0
   }
}

void compute_2(int n, double *y){
   for (int i=0; i<n; i++) {
      y[i] = 2.0;  // use 2.0
   }
}

/**
 Total the results and verify the results
*/
void compute_3(int n, double *y){
   double total=0.0;
   for (int i=0; i<n; i++) {
      total += y[i];
   };

   // expect the output to be the sum of (a * x[i] + y[i]) where
   // x[:] is initialized to 1.0, y[:] = 2.0

   if (total == (n*4.0)) {
      printf("PASS results are verified as correct\n");
   } else {
      printf("FAIL results are not correct. Expected %lf and received %lf. \n", (n*4.0), total);
   }
}

void daxpy_hip(int n, double a, double * x, double * y);

int main(int argc, char* argv[])
{
   int n = 1024;  // use 1024 for our example
   double a = 2.0;  // use 2.0 for our example
   double *x = new double[n];
   double *y = new double[n];
   printf("main Compiled with HOST_CODE \n");

   // allocate the device memory
   #pragma omp target data map(to:x[0:n]) map(tofrom:y[0:n])
   {
      compute_1(n, x);
      compute_2(n, y);
      #pragma omp target update to(x[0:n]) to(y[0:n]) // update x and y on the target
      #pragma omp target data use_device_ptr(x,y)
      {
         daxpy_hip(n, a, x, y);  // compute a * x[i] + y[i] in parallel
      }
   }
   compute_3(n, y);
}
#endif
