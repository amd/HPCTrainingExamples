// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// C OpenMP target-offload saxmy: y[i] = a*x[i]*y[i], the Part II companion to the
// HIP saxmy.cpp (a scaled Hadamard product, deliberately NOT a BLAS routine).
//
// THE BUG (the exercise): x is never allocated (left NULL), and the target region
// reads x[i] - the device dereferences a null pointer and faults. A malloc'd x
// would be silently demand-paged onto the MI300A even while unmapped, hiding the
// fault; a NULL x faults reliably, under any HSA_XNACK. The reference solution is
// saxmy_fixed.c (allocate, initialise and map x). See README.org. Build:
//   amdclang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyc saxmy.c
#include <stdio.h>
#include <stdlib.h>
int main(void)
{
    const int n = 256;
    double a = 2.0;
    double *x = NULL;                                 // BUG: x is never allocated
    double *y = (double*)malloc(n*sizeof(double));
    for (int i = 0; i < n; i++) y[i] = 2.0;
    #pragma omp target teams distribute parallel for map(tofrom:y[0:n])
    for (int i = 0; i < n; i++)
        y[i] = a*x[i]*y[i];                           // device reads x[i] from NULL -> fault
    printf("y[0] = %f  y[255] = %f\n", y[0], y[n-1]);
    free(y);
    return 0;
}
