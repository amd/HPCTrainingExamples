/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
This software is distributed under the MIT License

*/

#include "Science.hh"

#pragma omp declare target
void Science::compute(double *x, int N){
   *x = 1.0;
}
#pragma omp end declare target
