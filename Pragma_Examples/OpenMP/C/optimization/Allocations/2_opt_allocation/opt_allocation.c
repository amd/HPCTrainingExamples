// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <omp.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[])
{

#pragma requires unified_shared memory

   // Size of vectors
   int n = 10000000;
   int Niter = 10;
   double sum;

   double start_time = omp_get_wtime();

   // Allocate memory for each vector
   double *a = (double *) malloc(n*sizeof(double));
   double *b = (double *) malloc(n*sizeof(double));
   double *c = (double *) malloc(n*sizeof(double));

   for (int iter = 0; iter < Niter; iter++){
      // Initialize input vectors
      #pragma omp target teams loop
      for (int i = 0; i < n; i++){
         a[i] = sin((double)i)*sin((double)i);
         b[i] = cos((double)i)*cos((double)i);
         c[i] = 0.0;
      }

      #pragma omp target teams loop
      for (int i = 0; i < n; i++){
         c[i] = a[i] + b[i];
      }

      // Sum up vector c. Print result divided by n. It should equal 1
      sum = 0.0;
      #pragma omp target teams loop reduction(+:sum)
      for (int i = 0; i < n; i++){
          sum += c[i];
      }

      sum /= (double)n;
   }
 
   free(a);
   free(b);
   free(c);
    
   printf("Final result: %lf\n", sum);

   double end_time = omp_get_wtime();
   printf("Runtime is: %lf msecs\n",(end_time - start_time) * 1000.0);
}
