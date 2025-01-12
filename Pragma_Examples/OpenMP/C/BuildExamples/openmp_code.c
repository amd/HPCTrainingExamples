#pragma omp requires unified_shared_memory

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
   int M=100000;
   int Msize=M*sizeof(double);
   double sum=0.0;

   double* in_h = (double*)malloc(Msize);
   double* out_h = (double*)malloc(Msize);

   for (int i=0; i<M; i++) // initialize
      in_h[i] = 1.0;

#pragma omp target teams distribute parallel for map(to:in_h) map(from:out_h)
   for (int i=0; i<M; i++){
      out_h[i] = in_h[i] * 2.0;
   }

#pragma omp target teams distribute parallel for reduction(+:sum) map(to:out_h)
   for (int i=0; i<M; i++)
     sum += out_h[i];

   printf("Result is %lf\n",sum);

   free(in_h);
   free(out_h);
}
