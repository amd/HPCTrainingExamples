/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#pragma omp requires unified_shared_memory
int main(int argc, char *argv[]) {
  int n=10000;
  double ret = 0.0;
  double *b = (double*)malloc(n*sizeof(double));
  for(int i = 0; i < n; i++){
    b[i] = 1.0;
  }

  #pragma omp target teams distribute parallel for reduction(+:ret)
  for(int i = 0; i < n; i++) {
    #pragma omp atomic hint(AMD_fast_fp_atomics)
    ret += b[i];
  }

  if (ret != (double)n) {
     printf("FAILED: with sum %lf not equal to n %d\n",ret,n);
  } else {
     printf("Test PASSED\n");
  }

  free(b);
}
