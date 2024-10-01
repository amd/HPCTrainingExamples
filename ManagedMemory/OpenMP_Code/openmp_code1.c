#pragma omp requires unified_shared_memory

#include <stdio.h>
#include <stdlib.h>

void cpu_func(double *sum_out, double *out_h, double *in_h, int M){
   double sum;

#pragma omp target teams distribute parallel for simd
   for (int i=0; i<M; i++){
      out_h[i] = in_h[i] * 2.0;
   }

#pragma omp target teams distribute parallel for simd reduction(+:sum)
   for (int i=0; i<M; i++) // CPU-process
     sum += out_h[i];

   *sum_out = sum;
}

int main(int argc, char *argv[]) {
   int M=100000;
   int Msize=M*sizeof(double);
   double sum=0.0;

   double* in_h = (double*)malloc(Msize);
   double* out_h = (double*)malloc(Msize);

   for (int i=0; i<M; i++) // initialize
      in_h[i] = 1.0;

   cpu_func(&sum, out_h, in_h, M);

   printf("Result is %lf\n",sum);

   free(in_h);
   free(out_h);
}
