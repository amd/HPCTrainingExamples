#!/bin/bash

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

# Inline regression test for `target ... nowait`.
#
# The user-facing nowait example lives in
# Pragma_Examples/OpenMP/C/Nowait/. This test is a self-contained
# single-kernel timing harness that verifies the underlying property
# the multi-kernel pattern relies on: when `target ... nowait` is
# placed inside an OpenMP parallel region, the encountering thread
# returns from the kernel launch in microseconds rather than waiting
# for the GPU kernel to complete.

CC=${CC:-amdclang}
ROCM_GPU=$(rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]{1}' | sed -e 's/ *Name: *//')

case "$(basename $CC)" in
   amdclang*|clang*) OFFLOAD_FLAGS="--offload-arch=${ROCM_GPU}" ;;
   gcc*)             OFFLOAD_FLAGS="-foffload=-march=${ROCM_GPU}" ;;
   *)                OFFLOAD_FLAGS="" ;;
esac

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

cat > "$TMPDIR/test.c" <<'EOF'
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
   double t0, t_kernel, t_target_return = 0.0;
   int passed = 1;

   for (int i = 0; i < N; i++) { a[i] = (double)i * 1.0e-6; b[i] = 2.0; }

   /* warm-up */
   #pragma omp target teams distribute parallel for \
      map(to: a[0:N]) map(from: c_sync[0:N])
   for (int i = 0; i < N; i++) c_sync[i] = a[i];

   /* time the kernel synchronously */
   t0 = omp_get_wtime();
   #pragma omp target teams distribute parallel for \
      map(to: a[0:N], b[0:N]) map(from: c_sync[0:N])
   for (int i = 0; i < N; i++) {
      double v = a[i];
      for (int k = 0; k < K_GPU; k++) v = sin(v) + cos(v);
      c_sync[i] = v * b[i];
   }
   t_kernel = omp_get_wtime() - t0;

   /* same kernel inside parallel + masked, with nowait */
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
         t_target_return = omp_get_wtime() - tA;
      }
      #pragma omp for schedule(dynamic, 1024)
      for (int i = 0; i < N; i++) {
         double v = a[i];
         for (int k = 0; k < K_GPU/100; k++) v = sin(v) + cos(v);
         cpu_out[i] = v;
      }
   }

   for (int i = 0; i < N; i++) {
      if (fabs(c_sync[i] - c_async[i]) > 1.0e-12) { passed = 0; break; }
   }

   printf("kernel time alone           : %.4f s\n", t_kernel);
   printf("masked thread held at nowait: %.4f s\n", t_target_return);

   if (passed && t_target_return < 0.5 * t_kernel) {
      printf("PASS!\n");
   } else if (!passed) {
      printf("FAIL! (kernel results differ between sync and nowait)\n");
   } else {
      printf("FAIL! (masked thread did not skip past target nowait; "
             "%.4f s held on a %.4f s kernel)\n", t_target_return, t_kernel);
   }

   free(a); free(b); free(c_sync); free(c_async); free(cpu_out);
   return 0;
}
EOF

$CC -O3 -fopenmp $OFFLOAD_FLAGS -lm "$TMPDIR/test.c" -o "$TMPDIR/test"
OMP_NUM_THREADS=8 "$TMPDIR/test"
