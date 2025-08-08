#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>

/* Macro for checking GPU API return values */
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

int main() {
    // Matrix dimensions
    int M = 1024, N = 1024, K = 1024;

    // Host matrices
    double *h_A = (double *) malloc(sizeof(double) * M * K);
    double *h_B = (double *) malloc(sizeof(double) * K * N);
    double *h_C = (double *) malloc(sizeof(double) * M * N);

    srand(1);
    for(int i = 0; i < M*K; ++i)
    {
        h_A[i] = (double)rand();
    }
    for(int i = 0; i < K*N; ++i)
    {
        h_B[i] = (double)rand();
    }
    for(int i = 0; i < M*N; ++i)
    {
        h_C[i] = (double)rand();
    }

    // 1. Initialize rocBLAS
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    // 2. Allocate device memory
    double *d_A, *d_B, *d_C;
    hipCheck(hipMalloc(&d_A, sizeof(double) * M * K));
    hipCheck(hipMalloc(&d_B, sizeof(double) * K * N));
    hipCheck(hipMalloc(&d_C, sizeof(double) * M * N));

    // 3. Copy host data to device
    hipCheck(hipMemcpy(d_A, h_A, sizeof(double) * M * K, hipMemcpyHostToDevice));
    hipCheck(hipMemcpy(d_B, h_B, sizeof(double) * K * N, hipMemcpyHostToDevice));

    // 4. Perform DGEMM
    double alpha = 1.0, beta = 0.0; // Example alpha and beta
    CHECK_ROCBLAS_STATUS(
       rocblas_dgemm(handle,
                  rocblas_operation_none, rocblas_operation_none,
                  M, N, K,
                  &alpha,
                  d_A, M,
                  d_B, K,
                  &beta,
                  d_C, M)
       );

    // 5. Copy device data to host
    hipCheck(hipMemcpy(h_C, d_C, sizeof(double) * M * N, hipMemcpyDeviceToHost));

    // 6. Clean up
    CHECK_ROCBLAS_STATUS(rocblas_destroy_handle(handle));
    hipCheck(hipFree(d_A));
    hipCheck(hipFree(d_B));
    hipCheck(hipFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
