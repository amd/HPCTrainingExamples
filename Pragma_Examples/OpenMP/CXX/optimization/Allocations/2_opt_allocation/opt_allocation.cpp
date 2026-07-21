// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <omp.h>
#include <cmath>
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{

#pragma requires unified_shared memory

   // Size of vectors
   int n = 10000000;
   int Niter = 10;
   double sum;

   double start_time = omp_get_wtime();

   // Allocate memory for each vector
   double *a = new double[n];
   double *b = new double[n];
   double *c = new double[n];

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

   delete[] a;
   delete[] b;
   delete[] c;

   std::cout << "Final result: " << std::fixed << std::setprecision(6) << sum << std::endl;

   double end_time = omp_get_wtime();
   std::cout << "Runtime is: " << (end_time - start_time) * 1000.0 << " msecs" << std::endl;
}
