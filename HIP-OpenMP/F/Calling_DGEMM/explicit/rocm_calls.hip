// Copyright AMD 2024-2025, MIT License, contact Bob.Robey@amd.com
#include <rocblas/rocblas.h>
extern "C" {
    void omp_dgemm(void *ptr, int &modeA, int &modeB, int &m, int &n, 
            int &k, double &alpha, double *A, int &lda, 
            double *B, int &ldb, double &beta, double *C, int &ldc) {
        rocblas_handle *handle = (rocblas_handle *) ptr;
        rocblas_status status =rocblas_dgemm(*handle,(rocblas_operation)modeA,(rocblas_operation)modeB,m,n,k,
            &alpha,A,lda,B,ldb,&beta,C,ldc);

        // Check for errors
        if (status != rocblas_status_success) {
            fprintf(stderr, "rocblas_dgemm failed with status %d\n", status);
        }

    }
    void init_rocblas(void *ptr) {
        rocblas_handle *handle = (rocblas_handle *) ptr;
        rocblas_create_handle(handle);
    }
}
