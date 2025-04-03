#include <iostream>
#include "daxpy.hpp"
#include <vector>

int main(int argc, char* argv[]) {

   double a = 0.5;
   int N = 10;

   // allocate on host and device
   double* x = new double[N];
   double* y = new double[N];

   daxpy data(a,N);

   // initialize arrays
   #pragma omp target teams distribute parallel for
   for(int i=0; i<N; i++){
      data.setX(i,1.0);
      data.setY(i,0.5);
   }

   // initialize the daxpy class
   data.printArrays();

   // perform daxpy
   data.apply();

   data.printArrays();

   // free arrays on host
   free(x);
   free(y);

   return 0;

}
