#include <iostream>
#include "operations.hpp"
#include <cmath>

// require unified shared memory
#pragma omp requires unified_shared_memory

int main(int argc, char* argv[]) {

   // initialize the data
   int N = 10;
   double a = 0.5;
   std::string type = "L2";

   // operations constructor
   operations ops(a,N,type);

   // write member data with "set" functions
   #pragma omp target teams loop
   for(int i=0; i<N; i++){
      ops.setBoth(i,0.5);
      ops.daxpySetX(i,1.0);
   }

   ops.printAll();

   // compute daxpy operation using
   // member "get" and "set" functions   
   #pragma omp target teams distribute parallel for 
   for(int i=0; i<N; i++){
      double val = (ops.getConst() * ops.daxpyGetX(i) + ops.daxpyGetY(i)) / std::sqrt(N);
      ops.setBoth(i,val);
   }   

   // compute the norm of the vector
   // obtained with the above daxpy operation
   ops.updateNorm();
   
   ops.printAll();

   return 0;

}
