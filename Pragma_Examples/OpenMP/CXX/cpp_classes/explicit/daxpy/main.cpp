#include <iostream>
#include "daxpy.hpp"
#include <vector>

int main(int argc, char* argv[]) {

   double a = 0.5;
   int N = 10;

   // allocate on host and device
   daxpy data(a,N);

   // initialize arrays on GPU
   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      data.setX(i,1.0);
      data.setY(i,0.5);
   }

   data.updateHost();
   data.updateDevice();

   // perform daxpy with member function call
   //data.apply();

   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      double val = data.getConst() * data.getX(i) + data.getY(i);
      data.setY(i,val);
   }

   data.updateHost();
   data.printArrays();

   // free arrays on host
   free(x);
   free(y);

   return 0;

}
