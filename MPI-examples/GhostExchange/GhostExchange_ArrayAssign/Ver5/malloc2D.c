#include <stdlib.h>
#include "malloc2D.h"

double **malloc2D(int jmax, int imax, int joffset, int ioffset)
{
   // first allocate a block of memory for the row pointers and the 2D array
   double **x = (double **)malloc(jmax*sizeof(double *) + jmax*imax*sizeof(double));
#pragma omp target enter data map(alloc: x[:jmax+jmax*imax])

   // Now assign the start of the block of memory for the 2D array after the row pointers
   x[0] = (double *)(x + jmax + ioffset);

   // Last, assign the memory location to point to for each row pointer
   for (int j = 1; j < jmax; j++) {
      x[j] = x[j-1] + imax;
   }
   x += joffset;

#pragma omp target 
{
   // Do the same on the target
   // Now assign the start of the block of memory for the 2D array after the row pointers
   x[0] = (double *)(x + jmax + ioffset);

   // Last, assign the memory location to point to for each row pointer
   for (int j = 1; j < jmax; j++) {
      x[j] = x[j-1] + imax;
   }
}

   return(x);
}

void malloc2D_free(double **x, int joffset){
#pragma omp target exit data map(delete: x)
   x -= joffset;
   free(x);
}
