#include <stdio.h>
#include <omp.h>
//#include<memory>

void saxpy() {
   int N=1000000;
   float a, *x, *y;
   double t = 0.0;
   double tb, te;

   x = (float *)malloc(N*sizeof(float));
   y = (float *)malloc(N*sizeof(float));

   #pragma omp target enter data map(to:x[0:N],y[0:N])
   #pragma omp target teams distribute parallel for
   for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
   }
   a = 2.0f;

   tb = omp_get_wtime();

   #pragma omp target teams distribute parallel for
   for (int i = 0; i < N; i++) {
      y[i] = a * x[i] + y[i];
   }

   te = omp_get_wtime();
   t = te - tb;

   printf("Time of kernel: %lf\n", t);

   #pragma omp target update from(y[0:N])

   printf("plausibility check output:\n");
   printf("y[0] %lf\n",y[0]);
   printf("y[N-1] %lf\n",y[N-1]);
   
   #pragma omp target exit data map(release:x,y)

}
int main(int argc, char *argv[]){
   saxpy();
}
