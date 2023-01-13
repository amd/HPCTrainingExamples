
#include <stdio.h>
#include <stdlib.h>
#include "codelet.h"
#include <omp.h>

#define MAX(x,y) (x>y ? x : y)
#define MIN(x,y) ((x)<(y)?(x):(y))

#define NTIMERS 3

int main(int argc, char* argv[])
{
   int num_iteration=NTIMERS;   
   int n=100000;
   if (argc > 2) {
      int n=atoi(argv[1]);
   }   
   double *x = (double*)malloc(n*sizeof(double));
   double *y = (double*)malloc(n*sizeof(double));
   double sum=0;

   for (int i = 0; i < n; i++) {
        x[i] = 2.0f;
        y[i] = 1.0f;
   }

   
   double * timers = (double *)calloc(num_iteration,sizeof(double));
   for (int iter=0;iter<num_iteration; iter++)
   {
        double start = omp_get_wtime();

	sum=reduction(n, x, y);

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

   printf("Sum=%lf\n",sum);

   free(x);
   free(y);

   return 0;

}

