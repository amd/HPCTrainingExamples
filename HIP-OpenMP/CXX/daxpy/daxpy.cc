
#include <stdio.h>

// N must be a multiple of 256
#define N 1024
#define A 2

#ifdef __DEVICE_CODE__
#include <hip/hip_runtime.h>

__global__ void daxpy_kernel(int n, double a, double * x, double * y) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   y[i] = a * x[i] + y[i];
   // for debug
   printf("in kernel: y[%d]  is %g, a=%g, x[i]=%g \n", i, y[i], a, x[i]);
}

void daxpy_hip(int n, double a, double * x, double * y) {
   printf("daxpy_hip Compiled with DEVICE_CODE \n");
   assert(n % 256 == 0);
   double total=0.0;
   daxpy_kernel<<<n/256,256,0,NULL>>>(n, a, x, y);
   hipDeviceSynchronize();

   // for debug
   for (int i=0; i<n; i++)
   {
      total += y[i];
   };
   printf("in daxpy_hip: total for N=%d is %g \n", n, total);
}
#endif


#ifdef __HOST_CODE__

void compute_1(int n, double *x){
  // user defined
}

void compute_2(int n, double *y){
  // user defined
}

/**
 Total the results and verify the results
*/
void compute_3(int n, double *y){
   double total=0.0;
   for (int i=0; i<n; i++)
   {
      total += y[i];
   };

   // expect the output to be the sum of (2 * y[i]) where
   // y[i] is hardcoded to 1.0

   if (total == (N*2))
   {
           printf("PASS results are verified as correct\n");
   }
   else
   {
           printf("FAIL results are not correct. Expected %d and received %g. \n", (N*2), total);
   }
}


void daxpy_hip(int n, double a, double * x, double * y);

int main(int argc, char* argv[])
{
   //int n = 1000000;
   int n = N;     // use 1024 for our example
   double a = A;  // use 2 for our example
   double *x = new double[n];
   double *y = new double[n];
   printf("main Compiled with HOST_CODE \n");

   for (int i=0; i<n; i++)
   {
           x[i] = 1.0;  // use 1.0
   };

   // allocate the device memory
   #pragma omp target data map(to:x[0:count]) map(tofrom:y[0:count])
   {
      compute_1(n, x);
      compute_2(n, y);
      #pragma omp target update to(x[0:count]) to(y[0:count]) // update x and y on the target
      #pragma omp target data use_device_ptr(x,y)
      {
         daxpy_hip(n, a, x, y);  // compute 2 * y[i] in parallel
      }
   }
   compute_3(n, y);
}
#endif
