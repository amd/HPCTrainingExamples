#include <iostream>
#include "operations.hpp"
#include <cmath>

#pragma omp requires unified_shared_memory

int main(int argc, char* argv[]) {

   int N = 10;
   double a = 0.5;
   std::string type = "L2";
   operations ops(a,N,type);

   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      ops.setBoth(i,0.5);
      ops.daxpySetX(i,1.0);
   }

   ops.printAll();

   #pragma omp target teams distribute parallel for 
   for(int i=0; i<N; i++){
      double val = (ops.getConst() * ops.daxpyGetX(i) + ops.daxpyGetY(i)) / std::sqrt(N);
      ops.setBoth(i,val);
   }   

   ops.updateNorm();
   
   ops.printAll();

   return 0;

}
