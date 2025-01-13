#include <stdio.h>
#include <omp.h>

void saxpy(float a, float *x, float *y, int N) {
   double tb, te;

   tb = omp_get_wtime();
#pragma omp target teams loop
   for (int i = 0; i < N; i++) {
      y[i] += a * x[i];
   }
   te = omp_get_wtime();

   printf("Time of kernel: %lf\n", te - tb);

   printf("check output:\n");
   printf("y[0] %lf\n",y[0]);
   printf("y[N-1] %lf\n",y[N-1]);
}

int main(int argc, char *argv[]){
   int N=10000000;
   float a=2.0f;

   float *x = (float *)malloc(N*sizeof(float));
   float *y = (float *)malloc(N*sizeof(float));

   for (int i = 0; i < N; i++) {
      x[i] = 1.0f; y[i] = 2.0f;
   }

   saxpy(a, x, y, N);

   free(x); free(y);
}
