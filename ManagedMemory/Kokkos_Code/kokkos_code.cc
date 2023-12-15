#include <stdlib.h>
#include <stdio.h>
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[])
{
   int M=1000;
   int Msize=M*sizeof(double);
   double sum=0.0;

   Kokkos::initialize(argc, argv); {

   double* in_h = (double*)malloc(Msize);
   double* out_h = (double*)malloc(Msize);

   for (int i=0; i<M; i++) // initialize
      in_h[i] = 1.0;

   Kokkos::parallel_for(M, [=] (const  int i){ 
      out_h[i] = in_h[i] * 2.0;
   });

   Kokkos::fence();

   for (int i=0; i<M; i++) // CPU-process
     sum += out_h[i];

   printf("Result is %lf\n",sum);

   free(in_h);
   free(out_h);

   } Kokkos::finalize();
}
