/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

*/

// Author: Bob Robey, Bob.Robey@amd.com

#include <iostream>

#include "HotScience.hh"

using namespace std;

int main(int argc, char *argv[]){

   HotScience myscienceclass;

   int N=10000;
   double *x = new double[N];

   for (int k = 0; k < N; k++){
      myscienceclass.compute(&x[k], N);
   }

   cout << "Array value is " << x[0] << endl;
   cout << "Finished calculation" << endl;

   delete[] x;
}
