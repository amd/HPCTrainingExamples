// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

#include <omp.h>
#include <math.h>
#include <stdio.h>

#include "umpire/interface/c_fortran/umpire.h"

int main(int argc, char *argv[])
{

#pragma requires unified_shared memory

   // Size of vectors
   int n = 10000000;
   int Niter = 10;
   double sum;

   double start_time = omp_get_wtime();

   umpire_resourcemanager rm;
   umpire_resourcemanager_get_instance(&rm);

   umpire_allocator allocator;
   umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &allocator);

   umpire_allocator pool;
   umpire_resourcemanager_make_allocator_quick_pool(&rm, "pool", allocator, 1024*512, 512, &pool);

   for (int iter = 0; iter < Niter; iter++){
      // Allocate memory for each vector
      double *a = (double *) umpire_allocator_allocate(&pool, n*sizeof(double));
      double *b = (double *) umpire_allocator_allocate(&pool, n*sizeof(double));
      double *c = (double *) umpire_allocator_allocate(&pool, n*sizeof(double));

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
 
      umpire_allocator_deallocate(&pool, a);
      umpire_allocator_deallocate(&pool, b);
      umpire_allocator_deallocate(&pool, c);
   }
    
   printf("Final result: %lf\n", sum);

   double end_time = omp_get_wtime();
   printf("Runtime is: %lf msecs\n",(end_time - start_time) * 1000.0);
}
