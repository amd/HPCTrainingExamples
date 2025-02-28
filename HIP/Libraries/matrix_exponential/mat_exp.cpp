#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#ifdef __DEVICE_CODE__
#include <rocblas/rocblas.h>

extern void omp_dgemm(void *ptr, int op, int n, double alpha, double *A, int lda, int ldb, double beta, int ldc) {
        rocblas_handle *handle = (rocblas_handle *) ptr;
        rocblas_dgemm(*handle,(rocblas_operation)op,(rocblas_operation)op,n,n,n,&alpha,A,lda,A,ldb,&beta,A,ldc);
}

extern void init_rocblas(void *ptr) {
   rocblas_handle *handle = (rocblas_handle *) ptr;
   rocblas_create_handle(handle);
}

extern void finalize_rocblas(void *ptr){
   rocblas_handle *handle = (rocblas_handle *) ptr;
   rocblas_destroy_handle(*handle);
}

#endif

#ifdef __HOST_CODE__

void omp_dgemm(void *ptr, int op, int n, double alpha, double *A, int lda, int ldb, double beta, int ldc);

void init_rocblas(void *ptr);

void finalize_rocblas(void *ptr);

int main(int argc, char* argv[]) {

   // number of temrs in truncated series
   int N = 50;

   // allocate matrix A on host 
   double* A;
   A = (double*)malloc(4 * sizeof(double));

   // init A on host (row-major)
   A[0]=-2.0;
   A[1]=-1.0;
   A[2]=1.0;
   A[3]=-2.0;

   // initial solution
   std::vector<double> x_0(2);
   x_0[0]=1.0;
   x_0[1]=0.0;

   // rocblas dgemm parameters
   // dgemm performs C = alpha * op(A) * op(B) + beta * C
   double alpha = 1.0;
   double beta = 0.0;
   // set the modes so that op(A)=A and op(B)=B (could be a transpose otherwise)
   int op = 1;
   // specify leading dimensions
   int lda=2, ldb=2, ldc=2;
   void *ptr;
   init_rocblas(ptr);

   // use rocrand to set random time h_t
   // at which the error will be evaluated
//   rocrand_generator gen;
//   float mean = 0.0;
//   float stddev = 1.0;
//   float* d_t;
//   hipCheck( hipMalloc((void**)&d_t,sizeof(float)) );
//   rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
//   rocrand_set_seed(gen, time(NULL));
//   rocrand_generate_normal(gen, d_t, 1, mean, stddev);

   // exact solution vector evalated at h_t: x(h_t)
   std::vector<double> x_exact(2);
//   float t=*d_t;
     float t=1.0;
   x_exact[0]=exp(-2.0*t)*cos(t);
   x_exact[1]=exp(-2.0*t)*sin(t);


   // approximate matrix exponential
   double* EXP; 
   EXP = (double*)malloc(4 * sizeof(double));
   // initialize to first two summands
   // row-major
   EXP[0]=1.0 + A[0] * t;
   EXP[1]=A[1] * t;
   EXP[2]=A[2] * t;
   EXP[3]=1.0 + A[3] * t;

   double* tmp; 
   tmp = (double*)malloc(4 * sizeof(double));

#pragma omp target teams distribute parallel for map(to:A[:4]) map(alloc:tmp[:4]) reduction(+:EXP[:4]) 
   for(int i=2; i<N; i++){
      // init tmp where A^k will be computed
      tmp[0]=A[0];
      tmp[1]=A[1];
      tmp[2]=A[2];
      tmp[3]=A[3];
      int denom = i;
      float num = t;
      for(int k=1; k<i; k++){
         omp_dgemm(ptr, op, 2, alpha, tmp, lda, ldb, beta, ldc);
        // compute factorial;
        denom *= k;
        num *= t;
      }
      // reduciton to array step
      for(int i=1; i<4; i++){
         EXP[i]+=tmp[i] * num / denom;
      }
    }

   // compute approx solution
   std::vector<double> x_approx(2);
   x_approx[0] = EXP[0]*x_0[0] + EXP[1]*x_0[1];
   x_approx[1] = EXP[2]*x_0[0] + EXP[3]*x_0[1];

   // compute L2 norm of error \|x_exact-x_approx\|_2
   double norm = (x_exact[0] - x_approx[0])*(x_exact[0] - x_approx[0]);
   norm += (x_exact[1] - x_approx[1])*(x_exact[1] - x_approx[1]);
   norm = std::sqrt(norm);

   std::cout<<std::setprecision(16)<<"L2 norm of error is: " << norm << std::endl;

   free(A);
   free(EXP);
//   hipCheck( hipFree(d_t) );
//   rocrand_destroy_generator(gen);
   finalize_rocblas(ptr);

   return 0;

}

#endif

