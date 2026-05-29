/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

*/
#ifndef NO_UNIFIED_SHARED_MEMORY
#pragma omp requires unified_shared_memory
#endif

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <omp.h>

#define NTIMERS 1

using namespace std;

void daxpy(int n, double a, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z);

int main(int argc, char* argv[])
{
   int num_iteration=NTIMERS;   
   int n = 100000;
   double main_timer = 0.0;
   double main_start = omp_get_wtime();
   if (argc > 1) {
      n=atoi(argv[1]);
   }
   double a = 3.0;
   double *x = new (align_val_t(64) ) double[n];
   double *y = new (align_val_t(64) ) double[n];
   double *z = new (align_val_t(64) ) double[n];
#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n])

#pragma omp target teams distribute parallel for simd
   for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 1.0;
   }

   double * timers = (double *)calloc(num_iteration,sizeof(double));
   for (int iter=0;iter<num_iteration; iter++)
   {
        double start = omp_get_wtime();

        daxpy(n, a, x, y, z);

	timers[iter] = omp_get_wtime()-start;
   }

   double sum_time =  0.0;
   double max_time = -1.0e10;
   double min_time =  1.0e10;
   for (int iter=0; iter<num_iteration; iter++) {
        sum_time += timers[iter];
        max_time  = max(max_time,timers[iter]);
        min_time  = min(min_time,timers[iter]);
   }

   double avg_time = sum_time / (double)num_iteration;

   cout << "-Timing in Seconds: min=" << fixed << setprecision(6) << min_time << ", max=" <<max_time << ", avg=" << avg_time << endl;

   main_timer = omp_get_wtime()-main_start;
   cout << "-Overall time is " << main_timer << endl;
#pragma omp target update from(z[0])

   cout << "Last Value: z[" << n-1 << "]=" << z[n-1] << endl;

#pragma omp target exit data map(delete: x[0:n], y[0:n], z[0:n])
   delete [] x;
   delete [] y;
   delete [] z;

   return 0;
}

void daxpy(int n, double a, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
#pragma omp target teams distribute parallel for simd
        for (int i = 0; i < n; i++)
                z[i] = a*x[i] + y[i];
}
