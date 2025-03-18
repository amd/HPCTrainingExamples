#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <rocprofiler-sdk-roctx/roctx.h>

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
   int N = 20;
   // evaluating the solutions at t=0.5
   double t=0.5;

   // allocate matrices on host
   std::vector<double> h_A(4);
   std::vector<double> h_powA(4);

   // initialize matrices on host, in row-major
   // i=0 in the series
   h_A[0]=-2.0;
   h_A[1]=-1.0;
   h_A[2]=1.0;
   h_A[3]=-2.0;

   // i=1 in the series
   double h_EXP[4] = {1.0 + h_A[0] * t, h_A[1] * t, h_A[2] * t, 1.0 + h_A[3] * t};

   // init rocblas handle
   rocblas_handle handle;
   rocblas_create_handle(&handle);

   // exact solution vector evalated at t
   std::vector<double> x_exact(2);
   x_exact[0]=exp(-2.0*t)*cos(t);
   x_exact[1]=exp(-2.0*t)*sin(t);

#pragma omp parallel for reduction(+:h_EXP[:4]) firstprivate(h_powA) schedule(dynamic)
   for(int i=2; i<N; i++){
      omp_interop_t iobj = omp_interop_none;
      #pragma omp interop init(targetsync: iobj)
      hipStream_t stream = (hipStream_t) omp_get_interop_ptr(iobj, omp_ipr_targetsync, NULL);
      int tid = omp_get_thread_num();
      rocblas_set_stream(handle,stream);
      h_powA[0]=h_A[0];
      h_powA[1]=h_A[1];
      h_powA[2]=h_A[2];
      h_powA[3]=h_A[3];
      double denom = i;
      double num = t;
      for(int k=1; k<i; k++){
        roctxRangePush("rocblas_dgemm");
        rocblas_status status= rocblas_dgemm(handle,op,op,2,2,2,&alpha_dgemm,h_powA.data(),2,h_A.data(),2,&beta_dgemm,h_powA.data(),2);
        roctxRangePop();
	hipCheck( hipStreamSynchronize(stream) );
        // Check for errors
        if (status != rocblas_status_success) {
            fprintf(stderr, "rocblas_dgemm failed with status %d\n", status);
        }
        // compute factorial;
        denom *= k;
        num *= t;
      }
      #pragma omp interop destroy(iobj)
      // reduction to array step
      for(int m=0; m<4; m++){
         double factor = num / denom;
         h_EXP[m]+=h_powA[m] * factor;
      }
      if(i==2){
         int num_threads = omp_get_num_threads();
         std::cout << "Total num of threads is: " << num_threads << std::endl;
      }
    }

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

   if(norm < 1.0e-12){
      std::cout<<"PASSED!"<<std::endl;
      std::cout<<std::setprecision(16)<<"L2 norm of error is: " << norm << std::endl;
   }
   else{
      std::cout<<"FAILED!"<<std::endl;
      std::cout<<std::setprecision(16)<<"L2 norm of error is larger than prescribed tolerance..." << norm << std::endl;
   }

   rocblas_destroy_handle(handle);
   return 0;

}
