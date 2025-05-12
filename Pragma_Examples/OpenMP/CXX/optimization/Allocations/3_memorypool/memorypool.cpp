// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <omp.h>
#include <cmath>
#include <iostream>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"

int main(int argc, char *argv[])
{

#pragma requires unified_shared memory

   // Size of vectors
   int n = 10000000;
   int Niter = 10;
   double sum;

   double start_time = omp_get_wtime();

   auto& rm = umpire::ResourceManager::getInstance();

   auto allocator = rm.getAllocator("HOST");

   auto pooled_allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>("HOST_pool", allocator);

   float* my_data = static_cast<float*>(allocator.allocate(100*sizeof(float)));

   for (int iter = 0; iter < Niter; iter++){
      // Allocate memory for each vector
      double *a = static_cast<double *>(pooled_allocator.allocate(n*sizeof(double)));
      double *b = static_cast<double *>(pooled_allocator.allocate(n*sizeof(double)));
      double *c = static_cast<double *>(pooled_allocator.allocate(n*sizeof(double)));

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

      pooled_allocator.deallocate(a);
      pooled_allocator.deallocate(b);
      pooled_allocator.deallocate(c);
   }

   std::cout << "Final result: " << sum << std::endl;

   double end_time = omp_get_wtime();
   std::cout << "Runtime is: " << (end_time - start_time) * 1000.0 << " msecs" << std::endl;
}
