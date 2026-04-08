/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

*/

// OpenMP program to print Hello World
// using C language is supported by HIP

// HIP header
#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>

//OpenMP header
#include <omp.h>

#define NUM_THREADS 16
#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

__global__ void
hip_helloworld(unsigned omp_id, int* A_d)
{
    // Note: the printf command will only work if printf is enabled in your build.
    printf("Hello World... from HIP thread = %u\n", omp_id);

    A_d[omp_id] = omp_id;
}

int main(int argc, char* argv[])
{
    int* A_h, * A_d;
    size_t Nbytes = NUM_THREADS * sizeof(int);

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf("info: running on device %s\n", props.name);

    A_h = (int*)malloc(Nbytes);
    CHECK(hipMalloc(&A_d, Nbytes));
    for (int i = 0; i < NUM_THREADS; i++) {
        A_h[i] = 0;
    }
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    // Beginning of parallel region
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        fprintf(stderr, "Hello World... from OMP thread = %d\n",
               omp_get_thread_num());

        hipLaunchKernelGGL(hip_helloworld, dim3(1), dim3(1), 0, 0, omp_get_thread_num(), A_d);
    }
    // Ending of parallel region

    hipStreamSynchronize(0);
    CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    printf("Device Results:\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("  A_d[%d] = %d\n", i, A_h[i]);
    }

    printf ("PASSED!\n");

    free(A_h);
    CHECK(hipFree(A_d));
    return 0;
}
