/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

*/

#include "HotScience.hh"

#pragma omp declare target
void HotScience::compute(double *x, int N){
   *x = 5.0;
}
#pragma omp end declare target
