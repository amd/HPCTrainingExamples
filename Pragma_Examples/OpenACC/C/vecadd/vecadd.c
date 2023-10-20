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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

int main()
{
  int N = 100000;

  // Input arrays
  double a[N];
  double b[N];
  // Output arrays
  double c[N];

  double sum;
  double time_sum = 0.0;
  struct timespec tstart;

  cpu_timer_start(&tstart);

  for (int i=0; i<N; i++){
    a[i]=sin((double)(i+1))*sin((double)(i+1));
    b[i]=cos((double)(i+1))*cos((double)(i+1));
    c[i]=0.0;
  }

#pragma acc parallel loop copyin(a[0:N], b[0:N]) copyout(c[0:N])
  for (int j = 0; j< N; j++){
    c[j] = a[j] + b[j];
  }

  sum = 0.0;
  for (int i=0; i<N; i++){
    sum += c[i];
  }
  sum = sum/(double)N;

  time_sum += cpu_timer_stop(tstart);

  printf("Final result: %lf\n",sum);
  printf("Runtime is: %lf secs\n", time_sum);
}
