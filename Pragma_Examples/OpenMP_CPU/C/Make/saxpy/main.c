
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


   printf("Last Value: z[%d]=%lf\n",n-1,z[n-1]);

   free(x);
   free(y);
   free(z);
   return 0;

}

