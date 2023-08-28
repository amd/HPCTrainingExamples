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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "gpu_functions.h"

int main( int argc, char *argv[] ){

  printf( "GPU vector addition \n" );

  // Set the GPU device
  int device_id = 0;  
  set_device( device_id );

  
  const uint N = 256 * 256 * 256 * 32;
  printf( "N elements: %d \n", N );
  
  //  Allocate host arrays  
  double *h_a, *h_b, *h_c; 
  h_a = (double *) malloc( N*sizeof(double) );
  h_b = (double *) malloc( N*sizeof(double) );
  h_c = (double *) malloc( N*sizeof(double) );

  // Initialize host arrays
  for ( uint i=0; i<N; ++i ){
    h_a[i] = i;
    h_b[i] = i/4.f;
  }
  
  // Allocate device arrays and copy the data from the host 
  double *d_a, *d_b, *d_c;
  allocate_device_arrays( N, d_a, d_b, d_c );
  copy_host_to_device( N, h_a, h_b, d_a, d_b );

  // Perform the vector addition (c = a + b) on the GPU
  float kernel_time;
  kernel_time = gpu_vector_add( N, d_a, d_b, d_c );

  printf( "Kernel executed in %.2f milliseconds. \n", kernel_time);
  printf( "BW = %.1f GB/s. \n", 3*N*sizeof(double)/(kernel_time*1e-3)/(1024*1024*1024) );

  // Copy result from device to host
  copy_device_to_host( N, d_c, h_c );


  // Validate the results
  bool validation_passed = true;
  for ( int i=0; i<N; i++ ){
    if ( h_c[i] != ( h_a[i] + h_b[i] ) ){
      printf( "ERROR: Result doesn't match expected value: %f   %f \n", h_c[i], h_a[i] + h_b[i] );
      validation_passed = false;
    }
  }
  if (validation_passed ) printf( "Validation PASSED. \n");

  printf( "Finished \n" );

}
