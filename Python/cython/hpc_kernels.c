/*
 * hpc_kernels.c
 *
 * CPU computation kernels extracted from HPCTrainingExamples.
 * Each function mirrors the core loop from the original source
 * (with the main() and OpenMP/timing scaffolding stripped out so
 * the pure kernel can be called from Python via Cython).
 *
 * Original sources:
 *   cpu_func  - ManagedMemory/CPU_Code/cpu_code.c
 *   saxpy     - Pragma_Examples/OpenMP/C/1_saxpy/0_saxpy_portyourself/saxpy.c
 *   vecadd    - Pragma_Examples/OpenMP/C/3_vecadd/0_vecadd_portyourself/vecadd.c
 *   reduction - Pragma_Examples/OpenMP/C/2_reduction/0_reduction_portyourself/reduction.c
 */

#include "hpc_kernels.h"

/* cpu_func (ManagedMemory/CPU_Code/cpu_code.c) */
void cpu_func(double *in, double *out, int M) {
    for (int i = 0; i < M; i++) {
        out[i] = in[i] * 2.0;
    }
}

void saxpy(float a, float *x, float *y, int N) {
    for (int i = 0; i < N; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void vecadd(double *a, double *b, double *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

double reduction(double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum = sum + x[i];
    }
    return sum;
}
