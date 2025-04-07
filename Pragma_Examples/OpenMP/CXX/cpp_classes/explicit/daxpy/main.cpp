#include <iostream>
#include "daxpy.hpp"
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {

   double a = 0.5;
   int N = 10000000;

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

   // these two lines below are for debugging
   // data.updateHost();
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
