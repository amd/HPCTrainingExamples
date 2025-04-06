#include <iostream>
#include "daxpy.hpp"
#include <cmath>

// require unified shard memory
#pragma omp requires unified_shared_memory

int main(int argc, char* argv[]) {

   // initialize data
   int N = 10000000;
   double a = 0.5;

   // daxpy constructor
   daxpy data(a,N);

   // initialize daxpy data with "set" functions
   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      data.setX(i,1.0);
      data.setY(i,0.5);
   }

   // compute daxpy operation using 
   // member "get" and "set" functions
   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      double val = data.getConst() * data.getX(i) + data.getY(i);
      data.setY(i,val);
   }

   // the line below is for debugging
   // data.printArrays();

   double check = 0.0;
   #pragma omp target teams loop reduction(+:check)
   for(int i=0; i<N; i++){
      check += data.getY(i);
   }

   if (fabs(check - N) < 1.e-10) {
      std::cout<<"PASS!"<<std::endl;
   }
   else {
      std::cout<<"FAIL!"<<std::endl;
   }

   return 0;

}
