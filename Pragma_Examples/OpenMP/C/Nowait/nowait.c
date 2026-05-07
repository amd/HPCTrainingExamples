/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

Multi-kernel `target ... nowait` + `depend` example.

This follows the pattern from the OpenMP 6.0 examples document
(https://www.openmp.org/wp-content/uploads/openmp-examples-6.0.pdf):

   #pragma omp parallel
   #pragma omp single
   {
      // independent producers
      #pragma omp target ... nowait depend(out: a) ...
      for (...) a[i] = ...;

      #pragma omp target ... nowait depend(out: b) ...
      for (...) b[i] = ...;

      // consumer -- waits for both producers via depend(in: a, b)
      #pragma omp target ... nowait depend(in: a, b) depend(out: c) ...
      for (...) c[i] = a[i] + b[i];

      // host work -- runs concurrently with the GPU kernels
      ...

      #pragma omp taskwait
   }

This example is here to demonstrate that, with `nowait`, the host
thread does not block at the end of a `target` construct: it returns
immediately and is free to issue more kernels and run host code while
the GPU is still busy. NOTE: this example does not show kernel-to-kernel 
concurrency and makes the GPU kernels heavy to facilitate seeing the nowait on the CPU side.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N (1 << 20)

int main()
{
   double *a = (double*)malloc(N * sizeof(double));
   double *b = (double*)malloc(N * sizeof(double));
   double *c = (double*)malloc(N * sizeof(double));
   double host_sum;
   double t0, t_total;

   t0 = omp_get_wtime();

   #pragma omp parallel
   #pragma omp single
   {
      /* Kernel 1: produce a[i] = sin(i)^2.
         depend(out: a) declares this task as a producer of `a`. */
      #pragma omp target teams distribute parallel for nowait \
         depend(out: a[0:N]) \
         map(from: a[0:N])
      for (int i = 0; i < N; i++) {
         a[i] = sin((double)i) * sin((double)i);
      }

      /* Kernel 2: produce b[i] = cos(i)^2.
         No depend ordering with kernel 1, so the runtime is free to run
         them concurrently on the GPU if it can. */
      #pragma omp target teams distribute parallel for nowait \
         depend(out: b[0:N]) \
         map(from: b[0:N])
      for (int i = 0; i < N; i++) {
         b[i] = cos((double)i) * cos((double)i);
      }

      /* Kernel 3: consume a and b, produce c.
         depend(in: a, b) makes this task wait for kernels 1 and 2. */
      #pragma omp target teams distribute parallel for nowait \
         depend(in: a[0:N], b[0:N]) depend(out: c[0:N]) \
         map(to: a[0:N], b[0:N]) map(from: c[0:N])
      for (int i = 0; i < N; i++) {
         c[i] = a[i] + b[i];
      }

      /* Host work: independent of a, b, c, so the depend graph allows
         it to overlap with the GPU kernels. */
      double s = 0.0;
      for (long long n = 0; n < 5000000LL; n++) {
         s += sin((double)n);
      }
      host_sum = s;

      /* Wait for all deferred target tasks to complete before reading
         c[] outside the single block. */
      #pragma omp taskwait
   }

   t_total = omp_get_wtime() - t0;

   double cmin = c[0], cmax = c[0];
   for (int i = 0; i < N; i++) {
      if (c[i] < cmin) cmin = c[i];
      if (c[i] > cmax) cmax = c[i];
   }

   printf("c[0]    = %.6f\n", c[0]);
   printf("c[N-1]  = %.6f\n", c[N-1]);
   printf("min(c)  = %.6f\n", cmin);
   printf("max(c)  = %.6f\n", cmax);
   printf("(every c[i] should be 1.0: sin^2 + cos^2 = 1)\n");
   printf("host_sum (concurrent CPU work)  = %.6e\n", host_sum);
   printf("total elapsed time              = %.4f s\n", t_total);

   free(a); free(b); free(c);
   return 0;
}
