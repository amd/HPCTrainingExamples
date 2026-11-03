#ifndef _MALLOC_H
#define _MALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
void mem_exit(void);
double *dvector(int n);
double **malloc2D(int jmax, int imax);

#ifdef __cplusplus
}
#endif

#endif
