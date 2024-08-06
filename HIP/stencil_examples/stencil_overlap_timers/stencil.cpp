/*
Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdio.h>
#include "hip/hip_runtime.h"
#include <sys/time.h>

/* Macro for checking GPU API return values */
#define gpuCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

/* Number of elements in the array (without ghost zones) */
#define N (1024 * 1024 * 256)

/* Stencil settings */
#define STENCIL_RADIUS 3
#define STENCIL_SIZE 7

/* Number of threads per block - defined here to use in kernel */
#define THREADS_PER_BLOCK 256

/* --------------------------------------------------
Stencil kernel
-------------------------------------------------- */
__global__ void stencil(double *a_in, double *a_out){

    /* Define the global thread ID */
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < N){

        double sum = 0.0;
   
        /* Perform stencil operation */ 
        for (int j=-STENCIL_RADIUS; j<=STENCIL_RADIUS; j++){

            if ( ((id+j) >= 0) && ((id+j) < N) ){
                sum = sum + a_in[id + j];
            }
        }
    
        a_out[id] = sum / (double)STENCIL_SIZE;
    }
}

/* --------------------------------------------------
CPU stencil
-------------------------------------------------- */
void cpu_stencil(double *in, double* out){

    for (int i=0; i<N; i++){

        double sum = 0.0;

        for (int j=-STENCIL_RADIUS; j<=STENCIL_RADIUS; j++){
  
            if ( ((i+j) >=0) && ((i+j) < N) ){
                sum = sum + in[i+j];
            }
        }
        
        out[i] = sum / (double)STENCIL_SIZE;
    }
}

/* --------------------------------------------------
Main program
-------------------------------------------------- */
int main(int argc, char *argv[]){

    /* Number of bytes in array */
    size_t bytes = N * sizeof(double);

    // Create start/stop event objects and variable for elapsed time in ms
    hipEvent_t start_kernel, stop_kernel;
    gpuCheck( hipEventCreate(&start_kernel) );
    gpuCheck( hipEventCreate(&stop_kernel) );

    hipEvent_t start_h2d, stop_h2d;
    gpuCheck( hipEventCreate(&start_h2d) );
    gpuCheck( hipEventCreate(&stop_h2d) );

    hipEvent_t start_d2h, stop_d2h;
    gpuCheck( hipEventCreate(&start_d2h) );
    gpuCheck( hipEventCreate(&stop_d2h) );

    /* Allocate host buffers */
    double *h_A_in, *h_A_out, *h_A_out_cpu;
    gpuCheck( hipHostMalloc(&h_A_in, bytes) );
    gpuCheck( hipHostMalloc(&h_A_out, bytes) );
    gpuCheck( hipHostMalloc(&h_A_out_cpu, bytes) );

    for (int i=0; i<N; i++){

        h_A_in[i]      = (double)rand()/(double)RAND_MAX;
        h_A_out[i]     = 0.0;
        h_A_out_cpu[i] = 0.0;
    }

    /* Allocate device buffers */
    double *d_A_in, *d_A_out;
    gpuCheck( hipMalloc(&d_A_in, bytes) );
    gpuCheck( hipMalloc(&d_A_out, bytes) );

    /* Number of thread blocks in grid */
    int blk_in_grid = ceil( float(N) / THREADS_PER_BLOCK );

    /* Start timer */
    struct timeval start_total_time, stop_total_time, elapsed_total_time;
    struct timeval start_cpu_time, stop_cpu_time, elapsed_cpu_time;
    gettimeofday(&start_total_time, 0);

    gpuCheck( hipEventRecord(start_h2d, NULL) );

    /* Copy arrays from host to device */
    gpuCheck( hipMemcpyAsync(d_A_in, h_A_in, bytes, hipMemcpyHostToDevice, NULL) );
    gpuCheck( hipMemcpyAsync(d_A_out, h_A_out, bytes, hipMemcpyHostToDevice, NULL) );

    gpuCheck( hipEventRecord(stop_h2d, NULL) );
    gpuCheck( hipEventRecord(start_kernel, NULL) );

    /* Launch stencil kernel */
    stencil<<<blk_in_grid, THREADS_PER_BLOCK>>>(d_A_in, d_A_out);

    gpuCheck( hipEventRecord(stop_kernel, NULL) );
    gpuCheck( hipEventRecord(start_d2h, NULL) );

    /* Copy out-array from device to host */
    gpuCheck( hipMemcpyAsync(h_A_out, d_A_out, bytes, hipMemcpyDeviceToHost, NULL) );

    gpuCheck( hipEventRecord(stop_d2h, NULL) );
    gettimeofday(&start_cpu_time, 0);

    /* Run CPU-version of stencil */
    cpu_stencil(h_A_in, h_A_out_cpu);

    gettimeofday(&stop_cpu_time, 0);

    gpuCheck( hipDeviceSynchronize() );

    gettimeofday(&stop_total_time, 0);
    timersub(&stop_total_time, &start_total_time, &elapsed_total_time);
    timersub(&stop_cpu_time, &start_cpu_time, &elapsed_cpu_time);

    float kernel_time;
    gpuCheck( hipEventSynchronize(stop_kernel) );
    gpuCheck( hipEventElapsedTime(&kernel_time, start_kernel, stop_kernel) );

    float h2d_time;
    gpuCheck( hipEventSynchronize(stop_h2d) );
    gpuCheck( hipEventElapsedTime(&h2d_time, start_h2d, stop_h2d) );

    float d2h_time;
    gpuCheck( hipEventSynchronize(stop_d2h) );
    gpuCheck( hipEventElapsedTime(&d2h_time, start_d2h, stop_d2h) );

    /* Check results */
    double tolerance   = 1.0e-14;

    for (int i=0; i<N; i++){
        if ( fabs(h_A_out[i] - h_A_out_cpu[i]) > tolerance){
            printf("Error: h_A_out[%d] - h_A_out_cpu[%d] = %0.14f - %0.14f = %0.14f > %0.14f\n", i, i, h_A_out[i], h_A_out_cpu[i], fabs(h_A_out[i] - h_A_out_cpu[i]), tolerance);
            exit(1);
        }
    }

    /* Free host memory */
    gpuCheck( hipHostFree(h_A_in) );
    gpuCheck( hipHostFree(h_A_out) );
    gpuCheck( hipHostFree(h_A_out_cpu) );

    /* Free device memory */
    gpuCheck( hipFree(d_A_in) );
    gpuCheck( hipFree(d_A_out) );

    printf("__SUCCESS__\n");
    printf("Elapsed total time (ms): %0.2f\n", elapsed_total_time.tv_sec*1000.0+elapsed_total_time.tv_usec/1000.0);
    printf("Elapsed CPU time (ms): %0.2f\n", elapsed_cpu_time.tv_sec*1000.0+elapsed_cpu_time.tv_usec/1000.0);
    printf("Kernel time (ms): %0.2f\n", kernel_time);
    printf("H2D time (ms): %0.2f\n", h2d_time);
    printf("D2H time (ms): %0.2f\n", d2h_time);

    return 0;
}
