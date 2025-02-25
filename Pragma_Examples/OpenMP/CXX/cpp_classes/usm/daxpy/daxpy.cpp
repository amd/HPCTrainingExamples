#include<iostream>
#include "daxpy.hpp"

void daxpy::apply(){

   // note: we are not using this routine with usm
   // the daxpy operation is actually called from main
   // using member functions to access the private data

   #pragma omp target teams distribute parallel for 
   for(int i=0; i<N_; i++){
      y_[i] = a_ * x_[i] + y_[i];
   }

}
