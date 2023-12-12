#include <stdio.h>
#include <omp.h>

void saxpy(float a, float* x, float* y,
           int n) {

   int nteams = omp_get_num_teams();

   #pragma omp target teams map(to:x[0:n]) map(tofrom:y[0:n]) num_teams(nteams)
   {
      int bs = n / omp_get_num_teams(); // could also use nteams
      #pragma omp distribute
      for (int i = 0; i < n; i++) {
         #pragma omp parallel for simd firstprivate(i,bs)
         for (int ii = i; ii < i + bs; ii++) {
            y[ii] = a * x[ii] + y[ii];
         }
      }
   }
}
int main(int argc, char *argv[]){
   int N=1000000;
   float a, x[N], y[N];
   double tb, te;
   double t = 0.0;

   for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
   }

   tb = omp_get_wtime();

   saxpy(a, x, y, N);

   te = omp_get_wtime();
   t = te - tb;

   printf("Time of kernel: %lf\n", t);
}
