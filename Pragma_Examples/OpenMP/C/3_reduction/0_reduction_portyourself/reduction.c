/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
*/
#include <stdio.h>
#include <omp.h>

int main()
{
   int n=100000;
   double *x = (double*)malloc(n*sizeof(double));
   double sum=0.0;

   for (int i = 0; i < n; i++) {
        x[i] = 2.0;
   }
   
  double start = omp_get_wtime();

   for (int i = 0; i < n; i++) {
      sum = sum + x[i];
   }
  

   double time = omp_get_wtime()-start;
  

   printf("-Timing in Seconds: %f\n", time);

   printf("Sum=%lf\n",sum);

   free(x);

   return 0;

}
