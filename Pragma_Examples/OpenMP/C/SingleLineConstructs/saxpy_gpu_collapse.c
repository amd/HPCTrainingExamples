#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void saxpy(float a, float **x, float **y, int M, int N) {
   double tb, te;

   tb = omp_get_wtime();
   #pragma omp target teams distribute parallel for collapse(2)
   for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
         y[j][i] += a * x[j][i];
      }
   }
   te = omp_get_wtime();

   printf("Time of kernel: %lf\n", te - tb);

   printf("check output:\n");
   printf("y[0][0] = %lf\n",y[0][0]);
   printf("y[N-1][M-1] = %lf\n",y[N-1][M-1]);
}

float **malloc2D(int jmax, int imax)
{
   // first allocate a block of memory for the row pointers and the 2D array
   float **x = (float **)malloc(jmax*sizeof(float *) + jmax*imax*sizeof(float));

   // Now assign the start of the block of memory for the 2D array after the row pointers
   x[0] = (float *)(x + jmax);

   // Last, assign the memory location to point to for each row pointer
   for (int j = 1; j < jmax; j++) {
      x[j] = x[j-1] + imax;
   }

   return(x);
}

int main(int argc, char *argv[]){
   int N=1000,M=1000;
   float a=2.0f;

   float **x = (float **)malloc2D(N,M);
   float **y = (float **)malloc2D(N,M);

   for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
         x[j][i] = 1.0f; y[j][i] = 2.0f;
      }
   }

   saxpy(a, x, y, M, N);

   free(x); free(y);
}
