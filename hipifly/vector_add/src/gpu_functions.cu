/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "stdio.h"

#ifndef ENABLE_HIP
#include <cuda_runtime.h>
#else
// Translate CUDA functions to the analogous HIP function
#include "cuda_to_hip.h"
#endif

#define TPB 256 //Number of threads per block for kernel calls

void get_device_properties( const int device_id ){
 	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, device_id );
  printf("Device: %d  name:  %s \n", device_id, prop.name );

}


int set_device( int device_id ){
  
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  printf("Number of available devices %d\n", n_devices);  
  printf("Device id: %d \n", device_id);
  if ( device_id >= n_devices ){
    printf( "ERROR: Device %d is not available. Only %d devices detected.\n", device_id, n_devices );
    return -1;
  }
  
  cudaSetDevice(device_id); 
  get_device_properties( device_id );

  return 0;

}

void allocate_device_arrays( int N, double *&d_a, double *&d_b, double *&d_c  ){
  cudaMalloc( (void **)&d_a, N*sizeof(double) );
  cudaMalloc( (void **)&d_b, N*sizeof(double) );
  cudaMalloc( (void **)&d_c, N*sizeof(double) );  
}


void copy_host_to_device( int N, double *h_a, double *h_b, 
                          double *&d_a, double *&d_b   ){
  cudaMemcpy( d_a, h_a, N*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, h_b, N*sizeof(double), cudaMemcpyHostToDevice );
}


__global__ void vector_add_kernel( int N, double *d_a, double *d_b, double *d_c ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if ( tid < N ){
    d_c[tid] = d_a[tid] + d_b[tid];
  }
}

float gpu_vector_add( int N, double *d_a, double *d_b, double *d_c ){

  int n_grid = ( N - 1 )/TPB + 1;
  dim3 grid( n_grid, 1, 1 );
  dim3 block( TPB, 1, 1 );

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vector_add_kernel<<<grid, block, 0, 0>>>( N, d_a, d_b, d_c );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float elapsed_time_milliseconds = 0;
  cudaEventElapsedTime(&elapsed_time_milliseconds, start, stop);
  return elapsed_time_milliseconds;

}


void copy_device_to_host( int N, double *d_a, double *h_a ){
  cudaMemcpy( h_a, d_a, N*sizeof(double), cudaMemcpyDeviceToHost);
}
