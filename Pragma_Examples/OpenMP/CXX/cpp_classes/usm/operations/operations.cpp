#include<iostream>
#include "operations.hpp"

void operations::apply(){

  // note: we are not using this routine and
   // this operation is actually called from main
   // using member functions to access the private data

   d_.apply();
   #pragma omp target teams distribute parallel for
   for(int i=0; i<getSize(); i++){
      n_.setX(i,d_.getY(i));
   }
   n_.apply();
   n_.printNorm();

}
