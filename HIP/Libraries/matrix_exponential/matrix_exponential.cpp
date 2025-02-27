#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <rocrand/rocrand.h>
#include <rocblas/rocblas.h>
#include <cmath>

// Macro for checking GPU API return values 
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

int main(int argc, char* argv[]) {

   // number of temrs in truncated series
   int N = 1e4;

   // matrix A
   std::vector<double> h_A(4);
   double* d_A;
   hipCheck( hipMalloc(&d_A, 4*sizeof(double)) );
   // row-major
   h_A[0]=-2.0;
   h_A[1]=-1.0;
   h_A[2]=1.0;
   h_A[3]=-2.0;
   hipCheck( hipMemcpy(d_A, h_A.data(), 4*sizeof(double), hipMemcpyHostToDevice) );

   // rocblas dgemm parameters
   // dgemm performs C = alpha * A * B + beta * C
   double alpha = 1.0;
   double beta = 0.0;
   int lda,ldb,ldc;
   void* ptr;
   rocblas_handle *handle = (rocblas_handle *) ptr;
   rocblas_create_handle(handle);

   // use rocrand to set random time h_t
   // at which the error will be evaluated
   rocrand_generator gen;
   float mean = 0.0;
   float stddev = 1.0;
   float* h_t = new float[]{};
   float* d_t;
   hipCheck( hipMalloc((void**)&d_t,sizeof(float)) );
   rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
   rocrand_set_seed(gen, time(NULL));
   rocrand_generate_normal(gen, d_t, 1, mean, stddev);
std::cout<<"1 EEEEEEE"<<std::endl;
   hipCheck( hipMemcpy(&h_t, d_t, sizeof(float), hipMemcpyDeviceToHost) );

   // exact solution vector evalated at h_t: x(h_t)
   std::vector<double> x_exact(2);
std::cout<<"2 EEEEEEE"<<std::endl;
   float t=*h_t;
std::cout<<"2 bis EEEEEEE"<<std::endl;
   x_exact[0]=exp(-2.0*t)*cos(t);
   x_exact[1]=exp(-2.0*t)*sin(t);

std::cout<<"3 EEEEEEE"<<std::endl;

   // approximate matrix exponential
   std::vector<double> h_EXP(4);
   double* d_EXP;
   hipCheck( hipMalloc(&d_EXP, 4*sizeof(double)) );
   // initialize to first summand (the identity)
   // row-major
   h_EXP[0]=1.0;
   h_EXP[1]=0.0;
   h_EXP[2]=0.0;
   h_EXP[3]=1.0;


std::cout<<"4 EEEEEEE"<<std::endl;
// TODO: add loop to compute truncated series
//        rocblas_dgemm(*handle,(rocblas_operation)modeA,(rocblas_operation)modeB,2,2,2,
//            &alpha,d_EXP,lda,d_EXP,ldb,&beta,d_EXP,ldc);


std::cout<<"5 EEEEEEE"<<std::endl;
   hipCheck( hipFree(d_A) );
   free(h_t);
   hipCheck( hipFree(&d_t) );
   hipCheck( hipFree(d_EXP) );
   rocrand_destroy_generator(gen);

std::cout<<"6 EEEEEEE"<<std::endl;
   return 0;

}
