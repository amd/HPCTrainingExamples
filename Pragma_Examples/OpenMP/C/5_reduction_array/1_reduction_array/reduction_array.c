// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
// This software is distributed under the MIT License
//
// C version of reduction scalar reproducer created by Mahdieh Ghazimirsaeed
// Converted by Bob Robey

#include <stdio.h>

int main(int argc, char *argv[]){
   double ce[2]={0.0, 0.0};
#pragma omp target teams distribute parallel for simd reduction(+:ce[0:2])
   for (int j = 0; j< 1000; j++){
      ce[0] += 1.0;
      ce[1] += 1.0;
   }

   printf("ce[0] = %lf ce[1] = %lf\n", ce[0], ce[1]);
   return(0);
}
