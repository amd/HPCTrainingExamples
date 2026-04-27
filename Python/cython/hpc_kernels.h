/*
 * hpc_kernels.h
 *
 * Declarations for CPU computation kernels extracted from
 * HPCTrainingExamples so they can be compiled into a shared
 * library via Cython.
 *
 * Sources:
 *   cpu_func  – ManagedMemory/CPU_Code/cpu_code.c
 *   saxpy     – Pragma_Examples/OpenMP/C/1_saxpy/0_saxpy_portyourself/saxpy.c
 *   vecadd    – Pragma_Examples/OpenMP/C/3_vecadd/0_vecadd_portyourself/vecadd.c
 *   reduction – Pragma_Examples/OpenMP/C/2_reduction/0_reduction_portyourself/reduction.c
 */

#ifndef HPC_KERNELS_H
#define HPC_KERNELS_H

/* Double every element: out[i] = in[i] * 2.0 */
void cpu_func(double *in, double *out, int M);

/* SAXPY: y[i] = a * x[i] + y[i] */
void saxpy(float a, float *x, float *y, int N);

void vecadd(double *a, double *b, double *c, int N);

double reduction(double *x, int n);

#endif /* HPC_KERNELS_H */
