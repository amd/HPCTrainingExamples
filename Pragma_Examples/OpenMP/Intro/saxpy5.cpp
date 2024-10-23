#include <stdio.h>
#include <omp.h>

void saxpy(float a, float* x, float* y,
           int n) {
   double t = 0.0;
   double tb, te;

   tb = omp_get_wtime();

   #pragma omp target map(to:x[0:n]) \
                      map(tofrom:y[0:n])
   #pragma omp parallel for simd
   for (int i = 0; i < n; i++) {
      y[i] = a * x[i] + y[i];
   }

   te = omp_get_wtime();
   t = te - tb;

   printf("Time of kernel: %lf\n", t);

   if (y[0] > 1.0e30) {
      printf("y[0] %lf\n",y[0]);
   }
}
int main(int argc, char *argv[]){
   int N=1000000;
   float a, x[N], y[N];

   for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
   }

   saxpy(a, x, y, N);
}
