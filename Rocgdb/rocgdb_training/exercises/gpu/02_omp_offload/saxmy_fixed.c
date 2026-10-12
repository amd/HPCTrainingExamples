// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

// Reference solution for the C sibling of 02_omp_offload (compare with saxmy.c):
// allocate, initialise and map x so the device has a valid array to read. With
// a=2, x=1, y=2 the result is y=4.0, under any HSA_XNACK. See README.org. Build:
//   amdclang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyc_fixed saxmy_fixed.c
#include <stdio.h>
#include <stdlib.h>
int main(void)
{
    const int n = 256;
    double a = 2.0;
    double *x = (double*)malloc(n*sizeof(double));    // x allocated and initialised ...
    double *y = (double*)malloc(n*sizeof(double));
    for (int i = 0; i < n; i++) { x[i] = 1.0; y[i] = 2.0; }
    #pragma omp target teams distribute parallel for map(to:x[0:n]) map(tofrom:y[0:n])
    for (int i = 0; i < n; i++)                        // ... and mapped, so x[i] is valid
        y[i] = a*x[i]*y[i];
    printf("y[0] = %f  y[255] = %f\n", y[0], y[n-1]);
    free(x); free(y);
    return 0;
}
