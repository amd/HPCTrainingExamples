#include <iostream>
#include "daxpy.hpp"
#include <vector>

int main(int argc, char* argv[]) {

   double a = 0.5;
   int N = 10;

   // allocate on host and device
   double* x = new double[N];
   double* y = new double[N];
#pragma omp target enter data map(alloc:x[:N],y[:N])

   // initialize arrays on device
#pragma omp target teams loop
   for(int i=0; i<N; i++){
      x[i]=1.0;
      y[i]=0.5;
   }

   // initialize the daxpy class
   daxpy data(a,N,x,y);

#pragma omp target update from(x[:N],y[:N])
   data.printArrays();

   // perform daxpy
   data.apply();

#pragma omp target update from(x[:N],y[:N])
   data.printArrays();

   // free arrays on device and host
#pragma omp target exit data map(delete:x,y)
   free(x);
   free(y);

   return 0;

}
