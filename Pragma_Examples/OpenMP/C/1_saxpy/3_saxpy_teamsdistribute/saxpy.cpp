#include <stdio.h>
#include <omp.h>

#pragma omp requires unified_shared_memory

void saxpy() {
   int N=1000000;
   float a, x[N], y[N];
   double t = 0.0;
   double tb, te;


   #pragma omp target teams distribute parallel for simd
   for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
   }
   a = 2.0f;

   tb = omp_get_wtime();

   #pragma omp target teams distribute parallel for simd
   for (int i = 0; i < N; i++) {
      y[i] = a * x[i] + y[i];
   }

   te = omp_get_wtime();
   t = te - tb;

   printf("Time of kernel: %lf\n", t);
   
   printf("plausibility check output:\n");
   printf("y[0] %lf\n",y[0]);
   printf("y[N-1] %lf\n",y[N-1]);

}
int main(int argc, char *argv[]){
   saxpy();
}
