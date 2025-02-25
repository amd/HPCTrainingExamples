#include <iostream>
#include "daxpy.hpp"

#pragma omp requires unified_shared_memory

int main(int argc, char* argv[]) {

   int N = 10;
   double a = 0.5;
   daxpy data(a,N);

   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      data.setX(i,1.0);
      data.setY(i,0.5);
   }

   data.printArrays();

   #pragma omp target teams distribute parallel for 
   for(int i=0; i<N; i++){
      double val = data.getConst() * data.getX(i) + data.getY(i);
      data.setY(i,val);
   }   

   data.printArrays();

   return 0;

}
