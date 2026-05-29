// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
// This software is distributed under the MIT License
//
// This example was created by Johanna Potyka

#pragma omp requires unified_shared_memory
#pragma omp declare target
void compute(double *x){
   *x = 1.0;
}
#pragma omp end declare target
