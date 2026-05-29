// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
// This software is distributed under the MIT License
//
// This example was created by Johanna Potyka

#include <stdlib.h>

double *constants;

void initialize_constants(int isize){
   constants = (double *)malloc(isize*sizeof(double));
   for (int i = 0; i< isize; i++) {
      constants[i] = (double)i;
   }
}
