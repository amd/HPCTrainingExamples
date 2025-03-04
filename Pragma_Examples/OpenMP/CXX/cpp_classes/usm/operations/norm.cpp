#include<iostream>
#include "norm.hpp"
#include <cmath>

void norm::apply(){

    if (type_ == "L2"){
       #pragma omp target teams distribute parallel for reduction(+:norm_)
       for(int i=0; i<N_; i++){
          norm_+= (x_[i] * x_[i]); 
       }
       norm_=std::sqrt(norm_);
    }
    else if(type_ == "L1"){
       #pragma omp target teams distribute parallel for reduction(+:norm_)
       for(int i=0; i<N_; i++){
          norm_+= std::fabs(x_[i]); 
       }
    }
    else{
       std::cout<<"Norm " << type_ << " not yet implemented." <<std::endl;
    }

}
