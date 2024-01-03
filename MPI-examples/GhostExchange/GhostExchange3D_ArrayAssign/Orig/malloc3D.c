#include <stdlib.h>
#include <stdio.h>
#include "malloc3D.h"

double ***malloc3D(int knum, int jnum, int inum, int koffset, int joffset, int ioffset)
{
   double ***out;
   size_t mem_size;
   const size_t elsize = 8;

   mem_size  = knum*sizeof(void **);
   out       = (double ***)malloc(mem_size);

   mem_size  = knum*jnum*sizeof(void *);
   out[0]    = (double **) malloc(mem_size);

   size_t nelems = knum*jnum*inum;
   mem_size  = nelems*elsize;
   out[0][0] = (void *)calloc(nelems, elsize);

   for (int k = 0; k < knum; k++)
   {
      if (k > 0)
      {
         out[k] = out[k-1] + jnum;
         out[k][0] = out[k-1][0] + (jnum*inum);
      }

      for (int j = 1; j < jnum; j++)
      {
         out[k][j] = out[k][j-1] + inum;
      }
   }

   for (int k=0; k< knum; k++){
      for (int j=0; j< jnum; j++){
         out[k][j] += ioffset;
      }
   }
   for (int k=0; k< knum; k++){
      out[k] += joffset;
   }
   out += koffset;

   return (out);
}

void malloc3D_free(double ***var, int koffset, int joffset, int ioffset)
{
   var -= koffset;
   var[0] -= joffset;
   var[0][0] -= ioffset;

   free(var[0][0]);
   free(var[0]);
   free(var);
}




#ifdef XXX
double ***malloc3D(int kmax, int jmax, int imax, int koffset, int joffset, int ioffset)
{
   // first allocate a block of memory for the row pointers and the 2D array
   double ***x = (double ***)malloc(kmax*sizeof(double **) + kmax*jmax*sizeof(double *) + kmax*jmax*imax*sizeof(double));
//printf("DEBUG -- x %p file %s line %d\n",x,__FILE__,__LINE__);

   // Now assign the start of the block of memory for the 3D array after the row pointers
   x[0] = (double **)x + kmax*jmax; // + joffset;
//printf("DEBUG -- x[0] %p offset %d kmax %d jmax %d file %s line %d\n",x[0],(int)(x[0]-(double **)x),kmax,jmax,__FILE__,__LINE__);

   // Assign the memory location to point to for each row pointer
   for (int k = 0; k < kmax; k++) {
      if (k > 0){
         x[k] = x[k-1] + jmax*imax; // + joffset;
//printf("DEBUG -- x[%d] %p offset %d file %s line %d\n",k,x[k],x[k]-x[k-1],__FILE__,__LINE__);
         x[k][0] = x[k-1][0] + jmax; // + ioffset;
      }

      for (int j = 1; j < jmax; j++) {
         x[k][j] = x[k][j-1] + imax; // + ioffset;
//printf("DEBUG -- x[%d][%d] %p offset %d imax %d file %s line %d\n",k,j,x[k][j],x[k][j]-x[k][j-1],imax,__FILE__,__LINE__);
      }
   }
   //x += koffset;

   return(x);
}

void malloc3D_free(double ***x, int koffset, int joffset){
   x -= koffset;
   free(x);
}
#endif
