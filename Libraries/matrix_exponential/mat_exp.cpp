#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cmath>
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


// init rocblas parameters
// dgemm performs C = alpha_gemm * op(A) * op(B) + beta_gemm * C
const static double alpha_dgemm = 1.0;
const static double beta_dgemm = 0.0;
// set the modes so that op(A)=A and op(B)=B (could be a transpose otherwise)
const static rocblas_operation op = rocblas_operation_none;

int main(int argc, char* argv[]) {

   // number of terms in truncated series
   int N = 100;
   // evaluating the solutions at t=0.5
   double t=0.5;
   // temporary scalar for reduction
   double red=0.0;

   // allocate matrices on host
   std::vector<double> h_A(4);
   std::vector<double> h_EXP(4);
   std::vector<double> h_powA(4);

   // allocate matrices on device
   double* d_A;
   double*  d_powA;
   hipCheck( hipMalloc((void**)&d_A, 4*sizeof(double)) );
   // temporary matrix where the powers of A will be stored
   // while computing them inside the kernel
   hipCheck( hipMalloc(&d_powA, 4*sizeof(double)) );

   // initialize matrices on host, in row-major
   // i=0 in the series
   h_A[0]=-2.0;
   h_A[1]=-1.0;
   h_A[2]=1.0;
   h_A[3]=-2.0;

   // i=1 in the series
   h_EXP[0]=1.0 + h_A[0] * t;
   h_EXP[1]=h_A[1] * t;
   h_EXP[2]=h_A[2] * t;
   h_EXP[3]=1.0 + h_A[3] * t;

   // copy data from host to device
   hipCheck( hipMemcpy(d_A, h_A.data(), 4*sizeof(double), hipMemcpyHostToDevice) );

   // init rocblas handle
   rocblas_handle handle;
   rocblas_create_handle(&handle);

   // exact solution vector evalated at t
   std::vector<double> x_exact(2);
   x_exact[0]=exp(-2.0*t)*cos(t);
   x_exact[1]=exp(-2.0*t)*sin(t);

#pragma omp parallel for reduction(+:red) 
   for(int i=2; i<N; i++){
      // init d_powA on device
      hipCheck( hipMemcpy(d_powA, d_A, 4*sizeof(double), hipMemcpyDeviceToDevice) );
      double denom = i;
      double num = t;
      for(int k=1; k<i; k++){
        rocblas_dgemm(handle,op,op,2,2,2,&alpha_dgemm,d_powA,2,d_A,2,&beta_dgemm,d_powA,2);
        // compute factorial;
        denom *= k;
        num *= t;
      }
      hipCheck( hipMemcpy(h_powA.data(), d_powA, 4*sizeof(double), hipMemcpyDeviceToHost) );
      // reduction to array step
      for(int k=0; k<4; k++){
         double factor = num / denom;
         red+=h_powA[k] * factor;
      }
      if(i==2){
         int thread_id = omp_get_thread_num();
         int num_threads = omp_get_num_threads();
         std::cout << "thread ID is: " << thread_id << " total num of threads is: " << num_threads << std::endl;
      }
    }
    h_EXP[4]=red;

   // initial solution
   std::vector<double> x_0(2);
   x_0[0]=1.0;
   x_0[1]=0.0;

   // compute approx solution
   std::vector<double> x_approx(2);
   x_approx[0] = h_EXP[0]*x_0[0] + h_EXP[1]*x_0[1];
   x_approx[1] = h_EXP[2]*x_0[0] + h_EXP[3]*x_0[1];

   // compute L2 norm of error \|x_exact-x_approx\|_2
   double norm = (x_exact[0] - x_approx[0])*(x_exact[0] - x_approx[0]);
   norm += (x_exact[1] - x_approx[1])*(x_exact[1] - x_approx[1]);
   norm = std::sqrt(norm);

   if(norm < 1.0e12){
      std::cout<<"PASSED!"<<std::endl;
      std::cout<<std::setprecision(16)<<"L2 norm of error is: " << norm << std::endl;
   }
   else{
      std::cout<<"FAILED!"<<std::endl;
      std::cout<<std::setprecision(16)<<"L2 norm of error is larger than prescribed tolerance..." << norm << std::endl;
   }

   hipCheck( hipFree(d_A) );
   hipCheck( hipFree(d_powA) );
   rocblas_destroy_handle(handle);
   return 0;

}
