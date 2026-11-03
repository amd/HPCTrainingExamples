#include <stdio.h>
#include <stdlib.h>
void mem_exit(void);
double *dvector(int n)
{
  double *out;
  if ((out = (double *) calloc((unsigned) n, sizeof(double))) == NULL)
    mem_exit();
  return (out);
}
double **malloc2D(int jmax, int imax)
{
   // first allocate a block of memory for the row pointers and the 2D array
   double **x = (double **)malloc(jmax*sizeof(double *) + jmax*imax*sizeof(double));

   // Now assign the start of the block of memory for the 2D array after the row pointers
   x[0] = (double *)x + jmax;

   // Last, assign the memory location to point to for each row pointer
   for (int j = 1; j < jmax; j++) {
      x[j] = x[j-1] + imax;
   }

   return(x);
}
void mem_exit(void)
{
  puts("Allocation fault in memory routines.\n");
  exit(1);
}
