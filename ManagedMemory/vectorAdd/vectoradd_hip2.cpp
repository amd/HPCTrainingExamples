/*
Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void 
vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 

  {
 
      int x = blockDim.x * blockIdx.x + threadIdx.x;
      int y = blockDim.y * blockIdx.y + threadIdx.y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = b[i] + c[i];
      }

  }

using namespace std;

int main() {
  
  float *vectorA, *vectorB, *vectorC;

  int i, errors;

  vectorA = (float*)malloc(NUM * sizeof(float));
  vectorB = (float*)malloc(NUM * sizeof(float));
  vectorC = (float*)malloc(NUM * sizeof(float));
  
  //HIP_ASSERT(hipMalloc((void**)&vectorA, NUM * sizeof(float)));
  //HIP_ASSERT(hipMalloc((void**)&vectorB, NUM * sizeof(float)));
  //HIP_ASSERT(hipMalloc((void**)&vectorC, NUM * sizeof(float)));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    vectorB[i] = (float)i;
    vectorC[i] = (float)i*100.0f;
  }
  
  //HIP_ASSERT(hipMemcpy(vectorB, vectorB, NUM*sizeof(float), hipMemcpyHostToDevice));
  //HIP_ASSERT(hipMemcpy(vectorC, vectorC, NUM*sizeof(float), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(vectoradd_float, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  vectorA ,vectorB ,vectorC ,WIDTH ,HEIGHT);

  hipDeviceSynchronize();

  //HIP_ASSERT(hipMemcpy(vectorA, vectorA, NUM*sizeof(float), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (vectorA[i] != (vectorB[i] + vectorC[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }

  //HIP_ASSERT(hipFree(vectorA));
  //HIP_ASSERT(hipFree(vectorB));
  //HIP_ASSERT(hipFree(vectorC));

  free(vectorA);
  free(vectorB);
  free(vectorC);

  return errors;
}
