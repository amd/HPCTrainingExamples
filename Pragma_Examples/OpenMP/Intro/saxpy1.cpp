#include <stdio.h>
#include <omp.h>

void saxpy() {
   int N=1000000;
   float a, x[N], y[N];
   double t = 0.0;
   double tb, te;

   for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
   }

   tb = omp_get_wtime();

   #pragma omp target
   for (int i = 0; i < N; i++) {
      y[i] = a * x[i] + y[i];
   }

   te = omp_get_wtime();
   t = te - tb;

   printf("Time of kernel: %lf\n", t);
}
int main(int argc, char *argv[]){
   saxpy();
}
