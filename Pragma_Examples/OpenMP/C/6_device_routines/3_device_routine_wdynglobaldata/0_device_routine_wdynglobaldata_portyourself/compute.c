// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
// This software is distributed under the MIT License
//
// This example was created by Johanna Potyka


extern double *constants;

void compute(int cindex, double *x){
   *x = 1.0 + constants[cindex];
}

