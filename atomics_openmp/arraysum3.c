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

#pragma omp requires unified_shared_memory
int main(int argc, char *argv[]) {
  int errors=0, n=10000;
  double *a = (double*)malloc(n*sizeof(double));
  double *b = (double*)malloc(n*sizeof(double));
  for(int i = 0; i < n; i++){
    a[i] = 0.0;
    b[i] = 1.0;
  }

  #pragma omp target teams distribute parallel for map(tofrom: a[:n]) map(to: b[:n])
  for(int i = 0; i < n; i++) {
    a[i] += b[i];
  }

  for(int i = 0; i < n; i++) {
     if ( a[i] != b[i] ) {
        printf("Error -- i %d a %lf b %lf\n",i,a[i],b[i]);
        errors++;
     }
  }

  if (errors > 0) {
     printf("FAILED: with %d errors\n",errors);
  } else {
     printf("Test PASSED\n");
  }

  free(a);
  free(b);
}
