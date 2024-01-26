/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

This training example is released under the MIT license as listed
in the top-level directory. If this example is separated from the
main directory, include the LICENSE file with it.

Author: Gina Sitaraman
*/

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <sys/time.h>

#define ELAPSED(t1,t2) (t2.tv_sec-t1.tv_sec + (t2.tv_usec-t1.tv_usec)*1E-6)
#define NITER 25

#define HIP_CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
      }\
}

void print_arr (double *mat, int nelem)
{
    int i;
    for (i=0; i<nelem; i++) {
        printf ("%g ", mat[i]);
    }
    printf ("\n");
}

// Test results with host reference
int check_results (double *h_ref, double *h_z, size_t nelem)
{
    size_t i;
    for (i=0; i<nelem; i++) {
        if (h_ref[i] != h_z[i]) {
            printf ("ERROR at i=%lu, h_ref[i]=%f, h_z[i]=%f\n", i, h_ref[i], h_z[i]);
            return -1;
        }
    }
    printf ("PASSED\n");
    return 0;
}

__global__ void kernel_3 (double *d_x, double *d_y, double *d_z, double a, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    d_z[idx] = d_x[idx] * a + d_y[idx];
}

int main (int argc, char *argv[])
{
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);
    struct timeval t1, t2;

    if (argc < 2) {
        printf ("Usage: %s N where N is the length of arrays X, Y and Z\n", argv[0]);
        return -1;
    }
    size_t N = atol(argv[1]);

    size_t i, j, szbytes = N*sizeof(double);
    double *h_x = (double *) malloc (szbytes);
    double *h_y = (double *) malloc (szbytes);
    double *h_z = (double *) malloc (szbytes);
    double *h_ref = (double *) malloc (szbytes);
    double *d_x, *d_y, *d_z;
    double a = 1.0;

    // amt of data moved in GBytes
    float gbs = (float)(3*szbytes*NITER)/(1024.f*1024.f*1024.f); 

    // populate input arrays
    for (i=0; i<N; i++) {
        h_x[i] = (double)i;
        h_y[i] = (double)(i+1);
    }
    // populate h_ref
    for (i=0; i<N; i++) {
        h_ref[i] = h_x[i] * a + h_y[i];
    }
    //print_arr (h_ref, 10);

    // allocate GPU buffers
    HIP_CHECK(hipMalloc (&d_x, szbytes));
    HIP_CHECK(hipMalloc (&d_y, szbytes));
    HIP_CHECK(hipMalloc (&d_z, szbytes));

    // copy input arrays to GPU memory
    HIP_CHECK(hipMemcpy (d_x, h_x, szbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy (d_y, h_y, szbytes, hipMemcpyHostToDevice));

    //-----------------------------------------------------------------------------
    // Launch kernel that takes multiple blocks, mapping each thread to 1 array element
    //-----------------------------------------------------------------------------
    // Set up grid and block dimensions 
    dim3 grid_3((N+255)/256, 1, 1);
    dim3 block_3(256, 1, 1);

    // Launch kernel once first to warm up GPU and check correctness
    kernel_3 <<< grid_3, block_3 >>> (d_x, d_y, d_z, a, N);

    // copy result back to host buffer
    HIP_CHECK(hipMemcpy (h_z, d_z, szbytes, hipMemcpyDeviceToHost));
    check_results (h_ref, h_z, N);

    // Now measure bandwidth achieved
    gettimeofday (&t1, NULL);
    for (int iter=0; iter<NITER; iter++) {
        kernel_3 <<< grid_3, block_3 >>> (d_x, d_y, d_z, a, N);
    }
    hipDeviceSynchronize ();
    gettimeofday (&t2, NULL);
    printf ("daxpy_3: Bandwidth achieved = %.2f GB/s\n", gbs/ELAPSED(t1,t2));

    return 0;
}
