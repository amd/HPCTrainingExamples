#include <stdio.h>
#include <stdlib.h>

void cpu_func(double *in_h, double *out_h, int M){
   for (int i=0; i<M; i++){
      out_h[i] = in_h[i] * 2.0;
   }
}

int main(int argc, char *argv[]) {
   int M=1000;
   int Msize=M*sizeof(double);
   double sum=0.0;

   double* in_h = (double*)malloc(Msize);
   double* out_h = (double*)malloc(Msize);

   for (int i=0; i<M; i++) // initialize
      in_h[i] = 1.0;

      cpu_func(in_h, out_h, M);

   for (int i=0; i<M; i++) // CPU-process
     sum += out_h[i];

   printf("Result is %lf\n",sum);

   free(in_h);
   free(out_h);
}
