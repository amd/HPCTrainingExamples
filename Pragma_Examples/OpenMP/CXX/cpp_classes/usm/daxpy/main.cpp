#include <iostream>
#include "daxpy.hpp"

// require unified shard memory
#pragma omp requires unified_shared_memory

int main(int argc, char* argv[]) {

   // initialize data
   int N = 10;
   double a = 0.5;

   // daxpy constructor
   daxpy data(a,N);

   // initialize daxpy data with "set" functions
   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      data.setX(i,1.0);
      data.setY(i,0.5);
   }

   data.printArrays();

   // compute daxpy operation using 
   // member "get" and "set" functions
   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      double val = data.getConst() * data.getX(i) + data.getY(i);
      data.setY(i,val);
   }   

   data.printArrays();

   return 0;

}
