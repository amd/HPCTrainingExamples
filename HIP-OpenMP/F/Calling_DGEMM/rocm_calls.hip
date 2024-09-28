#include <rocblas/rocblas.h>
extern "C" {
    void omp_dgemm(void *ptr, int modeA, int modeB, int m, int n, 
            int k, double alpha, double *A, int lda, 
            double *B, int ldb, double beta, double *C, int ldc) {
        rocblas_handle *handle = (rocblas_handle *) ptr;
        rocblas_dgemm(*handle,(rocblas_operation)modeA,(rocblas_operation)modeB,m,n,k,
            &alpha,A,lda,B,ldb,&beta,C,ldc);
    }
    void init_rocblas(void *ptr) {
        rocblas_handle *handle = (rocblas_handle *) ptr;
        rocblas_create_handle(handle);
    }
}
