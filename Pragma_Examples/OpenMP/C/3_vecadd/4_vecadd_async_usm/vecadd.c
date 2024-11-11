/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

*/
#include <stdio.h>
//#include <stdlib.h>
#include <omp.h>
#include <math.h>


#pragma omp requires unified_shared_memory

int main()
{
  int N = 100000;

  // Input arrays
  double a[N];
  double b[N];
  // Output arrays
  double c[N];

  double sum;

  #pragma omp target teams distribute parallel for nowait depend(out:a)
  for (int i=0; i<N; i++){
    a[i]=sin((double)(i+1))*sin((double)(i+1));
  }
  #pragma omp target teams distribute parallel for nowait depend(out:b)
  for (int i=0; i<N; i++){
    b[i]=cos((double)(i+1))*cos((double)(i+1));
  }
  #pragma omp target teams distribute parallel for nowait depend(out:c)
  for (int i=0; i<N; i++){
    c[i]=0.0;
  }
  double tstart = omp_get_wtime();
  #pragma omp target teams distribute parallel for nowait depend(in:a,b) depend(inout:c)
  for (int j = 0; j< N; j++){
    c[j] = a[j] + b[j];
  }

  sum = 0.0;
  #pragma omp target teams distribute parallel for reduction(+: sum) nowait depend(in:c)
  for (int i=0; i<N; i++){
    sum += c[i];
  }
  #pragma omp taskwait

  sum = sum/(double)N;
  printf("Runtime is: %lf secs\n", omp_get_wtime()-tstart);

  printf("Final result: %lf\n",sum);

}
