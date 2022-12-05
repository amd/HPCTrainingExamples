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
#include <unistd.h>
#include "codelet.h"
#include <omp.h>

#define MAX(x,y) (x>y ? x : y)
#define MIN(x,y) ((x)<(y)?(x):(y))

#define NTIMERS 3

int main(int argc, char* argv[])
{
   int num_iteration=NTIMERS;   
   int n = 100000;
   if (argc > 1) {
      n=atoi(argv[1]);
   }
   float a = 3.0f;
   float *x = (float*)malloc(n*sizeof(float));
   float *y = (float*)malloc(n*sizeof(float));
   float *z = (float*)malloc(n*sizeof(float));


   for (int i = 0; i < n; i++) {
        x[i] = 2.0f;
        y[i] = 1.0f;
   }


   double * timers = (double *)calloc(num_iteration,sizeof(double));
   for (int iter=0;iter<num_iteration; iter++)
   {

        double start = omp_get_wtime();

        saxpy(n, a, x, y, z);

	timers[iter] = omp_get_wtime()-start;

   }


   double sum_time =  0.0;
   double max_time = -1.0e10;
   double min_time =  1.0e10;
   for (int iter=0; iter<num_iteration; iter++) {
        sum_time += timers[iter];
        max_time  = MAX(max_time,timers[iter]);
        min_time  = MIN(min_time,timers[iter]);
   }


   double avg_time = sum_time / num_iteration;

   printf("-Timing in Seconds: min=%f, max=%f, avg=%f\n", min_time, max_time, avg_time);


   printf("Last Value: y[%d]=%lf\n",n-1,y[n-1]);

   free(x);
   free(y);

   return 0;

}
