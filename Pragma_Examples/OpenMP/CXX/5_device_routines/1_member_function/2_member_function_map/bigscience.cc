// Copyright (c) 2025 AMD HPC Application Performance Team
// Author: Bob Robey, Bob.Robey@amd.com
// MIT License

#include <iostream>

#include "Science.hh"

using namespace std;

int main(int argc, char *argv[]){

   Science myscienceclass;

   int N=10000;
   double *x = new double[N];

#pragma omp target teams loop map(from:x[0:N])
   for (int k = 0; k < N; k++){
      myscienceclass.compute(&x[k], N);
   }
    
   cout << "Last x value: " << x[N-1] << endl;
   delete[] x;

   cout << "Finished calculation" << endl;
}
