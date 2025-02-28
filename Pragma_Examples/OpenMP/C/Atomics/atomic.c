/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
*/

#include <stdio.h>
#include <omp.h>
#include <math.h>

#pragma omp requires unified_shared_memory

int main()
{
  int N = 100000;
  double sum;

  // Input arrays
  double a[N];

  #pragma omp target teams distribute parallel for
  for (int i=0; i<N; i++){
    a[i]=1.0;
  }
  double tstart = omp_get_wtime();
  sum = 0.0;
  #pragma omp target teams distribute parallel for map(tofrom:sum)
  for (int i = 0; i< N; i++){
    #pragma omp atomic
    sum += a[i];
  }
  printf("   Atomic result: %lf Runtime is: %lf secs\n", sum, omp_get_wtime()-tstart);

  tstart = omp_get_wtime();
  sum = 0.0;
  #pragma omp target teams distribute parallel for reduction(+: sum)
  for (int i=0; i<N; i++){
    sum += a[i];
  }
  printf("Reduction result: %lf Runtime is: %lf secs\n", sum, omp_get_wtime()-tstart);
}
