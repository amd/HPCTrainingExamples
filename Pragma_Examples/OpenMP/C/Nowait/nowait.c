/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

Test that the OpenMP `nowait` clause on a
`target teams distribute parallel for` construct actually allows the
encountering thread to continue past the kernel launch BEFORE the GPU
kernel completes, so that the host can do other work while the GPU is
still running.

Pattern (from the OpenMP 6.0 spec, "nowait clause" example):

   #pragma omp parallel
   {
      #pragma omp masked
      {
         #pragma omp target teams distribute parallel for nowait \
            map(to: ...) map(from: ...)
         for (...) {  // GPU work
            ...
         }
      }

      #pragma omp for schedule(dynamic, chunk)
      for (...) {  // CPU work, masked thread joins here
         ...
      }
   }

Strategy
--------
1. Time the same GPU kernel run synchronously (no nowait) to obtain
   `t_kernel`.
2. Run the spec-style pattern above and measure how long the masked
   thread is held at the `target ... nowait` line, `t_target_return`.
3. PASS iff the GPU kernel produced the same output as the sync run AND
   `t_target_return` is much smaller than `t_kernel` (the masked thread
   came back well before the kernel completed).
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N      (1 << 20)
#define K_GPU  20000

int main()
{
   double *a       = (double*)malloc(N * sizeof(double));
   double *b       = (double*)malloc(N * sizeof(double));
   double *c_sync  = (double*)malloc(N * sizeof(double));
   double *c_async = (double*)malloc(N * sizeof(double));
   double *cpu_out = (double*)malloc(N * sizeof(double));
   double t0, t_kernel, t_target_return, t_total;
   int passed = 1;

   for (int i = 0; i < N; i++) { a[i] = (double)i * 1.0e-6; b[i] = 2.0; }

   /* Warm-up to absorb device-init cost */
   #pragma omp target teams distribute parallel for map(to: a[0:N]) map(from: c_sync[0:N])
   for (int i = 0; i < N; i++) c_sync[i] = a[i];

   /* ---- 1. Calibrate: time the kernel alone (synchronous) ---- */
   t0 = omp_get_wtime();
   #pragma omp target teams distribute parallel for \
      map(to: a[0:N], b[0:N]) map(from: c_sync[0:N])
   for (int i = 0; i < N; i++) {
      double v = a[i];
      for (int k = 0; k < K_GPU; k++) v = sin(v) + cos(v);
      c_sync[i] = v * b[i];
   }
   t_kernel = omp_get_wtime() - t0;

   /* ---- 2. Spec-style pattern: parallel + masked + target nowait ---- */
   t_target_return = 0.0;
   t0 = omp_get_wtime();
   #pragma omp parallel
   {
      #pragma omp masked
      {
         double tA = omp_get_wtime();
         #pragma omp target teams distribute parallel for nowait \
            map(to: a[0:N], b[0:N]) map(from: c_async[0:N])
         for (int i = 0; i < N; i++) {
            double v = a[i];
            for (int k = 0; k < K_GPU; k++) v = sin(v) + cos(v);
            c_async[i] = v * b[i];
         }
         double tB = omp_get_wtime();
         /* If `nowait` is honored, tB-tA is microseconds, not the
            full kernel time. */
         t_target_return = tB - tA;
      }

      /* All threads (including the masked one, after it falls through)
         participate in this CPU loop while the GPU kernel is in flight. */
      #pragma omp for schedule(dynamic, 1024)
      for (int i = 0; i < N; i++) {
         double v = a[i];
         for (int k = 0; k < K_GPU / 100; k++) v = sin(v) + cos(v);
         cpu_out[i] = v;
      }
   } /* implicit barrier here also waits for the deferred target task */
   t_total = omp_get_wtime() - t0;

   /* ---- 3. Correctness: same kernel, same inputs => identical output ---- */
   for (int i = 0; i < N; i++) {
      if (fabs(c_sync[i] - c_async[i]) > 1.0e-12) { passed = 0; break; }
   }

   printf("Calibrated kernel time alone           : %.4f s\n", t_kernel);
   printf("masked thread held at `target nowait`  : %.4f s\n", t_target_return);
   printf("Total parallel region time             : %.4f s\n", t_total);

   if (passed && t_target_return < 0.5 * t_kernel) {
      printf("PASS!\n");
   } else if (!passed) {
      printf("FAIL! (kernel results differ between sync and nowait variants)\n");
   } else {
      printf("FAIL! (masked thread held %.4f s on a %.4f s kernel; "
             "nowait was not honored)\n", t_target_return, t_kernel);
   }

   free(a); free(b); free(c_sync); free(c_async); free(cpu_out);
   return 0;
}
