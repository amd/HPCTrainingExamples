#include <iostream>
#include <vector>
#include <rocsparse/rocsparse.h>
#include <rocrand/rocrand.h>
#include <rocblas/rocblas.h>
#include <cmath>

int main(int argc, char* argv[]) {

   // number of temrs in truncated series
   int N = 1e4;

   // matrix A
   std::vector<double> h_A(4);
   double* d_A;
   hipMalloc(&d_A, 4*sizeof(double));
   // row-major
   h_A[0]=-2.0;
   h_A[1]=-1.0;
   h_A[2]=1.0;
   h_A[3]=-2.0;
   hipMemcpy(d_A, h_A, 4* sizeof(double), hipMemcpyHostToDevice);

   // rocblas dgemm parameters
   // dgemm performs C = alpha * A * B + beta * C
   double alpha = 1.0;
   double beta = 0.0;
   int lda,ldb,ldc;
   rocblas_handle *handle = (rocblas_handle *) ptr;
   rocblas_create_handle(handle);

   // use rocrand to set random time h_t
   // at which the error will be evaluated
   rocrand_generator gen;
   double* h_t, d_t;
   h_t = (double *)malloc(sizeof(double));
   hipMalloc(&d_t,sizeof(double));
   rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
   rocrand_set_seed(gen, time(NULL));
   rocrand_generate_uniform(gen, d_t, 1);
   hipMemcpy(h_t, d_t, sizeof(double), hipMemcpyDeviceToHost);

   // exact solution vector evalated at h_t: x(h_t)
   std::vector<double> x_exact(2);
   x_exact[0]=exp(-2.0*h_t[0])*cos(h_t[0]);
   x_exact[1]=exp(-2.0*h_t[0])*sin(h_t[0]);

   // approximate matrix exponential
   std::vector<double> h_EXP(4);
   double* d_EXP;
   hipMalloc(&d_EXP, 4*sizeof(double));
   // initialize to first summand (the identity)
   // row-major
   h_EXP[0]=1.0;
   h_EXP[1]=0.0;
   h_EXP[2]=0.0;
   h_EXP[3]=1.0;


// TODO: add loop to compute truncated series
//        rocblas_dgemm(*handle,(rocblas_operation)modeA,(rocblas_operation)modeB,2,2,2,
//            &alpha,d_EXP,lda,d_EXP,ldb,&beta,d_EXP,ldc);


   hipFree(d_A);
   free(h_t);
   hipFree(d_t);
   hipFree(d_EXP);
   rocrand_destroy_generator(gen);

   return 0;

}
