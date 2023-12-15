#include <stdio.h>
#include <stdlib.h>
#include "RAJA/RAJA.hpp"

using namespace std;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
   int M=1000;
   int Msize=M*sizeof(double);
   double sum=0.0;

   double* in_h = (double*)malloc(Msize);
   double* out_h = (double*)malloc(Msize);

   using EXEC_POL = RAJA::hip_exec<256>;

   for (int i=0; i<M; i++) // initialize
      in_h[i] = 1.0;

   RAJA::forall< EXEC_POL > (RAJA::RangeSegment(0, M), [=] (int i) {
      out_h[i] = in_h[i] * 2.0;
   } );

   for (int i=0; i<M; i++) // CPU-process
     sum += out_h[i];

   printf("Result is %lf\n",sum);

   free(in_h);
   free(out_h);
}
