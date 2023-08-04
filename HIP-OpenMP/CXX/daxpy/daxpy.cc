
#ifdef __DEVICE_CODE__
#include <hip/hip_runtime.h>

__global__ void daxpy_kernel(int n, double a, double * x, double * y) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   y[i] = a * x[i] + y[i];
}

void daxpy_hip(int n, double a, double * x, double * y) {
   assert(n % 256 == 0);
   daxpy_kernel<<<n/256,256,0,NULL>>>(n, a, x, y);
   hipDeviceSynchronize();
}
#endif

#ifdef __HOST_CODE__

void compute_1(int n, double *x){
}

void compute_2(int n, double *y){
}

void compute_3(int n, double *y){
}
void daxpy_hip(int n, double a, double * x, double * y);

int main(int argc, char* argv[])
{
   int n = 1000000;
   double a = 2.0;
   double *x = new double[n];
   double *y = new double[n];

   // allocate the device memory
   #pragma omp target data map(to:x[0:count]) map(tofrom:y[0:count])
   {
      compute_1(n, x); // mapping table: x:[0xabcd,0xef12], x = 0xabcd
      compute_2(n, y);
      #pragma omp target update to(x[0:count]) to(y[0:count]) // update x and y on the target
      #pragma omp target data use_device_ptr(x,y)
      {
         daxpy_hip(n, a, x, y); // mapping table: x:[0xabcd,0xef12], x = 0xef12
      }
   }
   compute_3(n, y);
}
#endif
