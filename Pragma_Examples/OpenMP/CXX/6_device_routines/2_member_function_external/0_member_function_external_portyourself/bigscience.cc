/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

*/

// Author: Bob Robey, Bob.Robey@amd.com

#include <iostream>

#include "Science.hh"

using namespace std;

int main(int argc, char *argv[]){

   Science myscienceclass;

   int N=10000;
   double *x = new double[N];

   for (int k = 0; k < N; k++){
      myscienceclass.compute(&x[k], N);
   }

   cout << "Last x value: " << x[N-1] << endl;
   delete[] x;

   cout << "Finished calculation" << endl;
}
