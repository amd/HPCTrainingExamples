/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

*/
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
   double *x = new double[n];
   double *y = new double[n];
   double *z = new double[n];

   for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 1.0;
   }

#pragma omp target enter data map(to: x[0:n], y[0:n]) map(alloc: z[0:n])

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

#pragma omp target exit data map(from: z[0:n]) map(delete: x[0:n], y[0:n])

   cout << "Last Value: z[" << n-1 << "]=" << z[n-1] << endl;

   delete [] x;
   delete [] y;
   delete [] z;

   return 0;
}

void daxpy(int n, double a, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
#pragma omp target teams distribute parallel for simd map(to: x[0:n], y[0:n]) map(from: z[0:n])
        for (int i = 0; i < n; i++)
                z[i] = a*x[i] + y[i];
}
