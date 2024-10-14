/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * This training example is released under the MIT license as listed
 * in the top-level directory. If this example is separated from the
 * main directory, include the LICENSE file with it.
 *
 * Author: Carlo Bertolli
 * */
#include <cstdio>

int main()
{
  const size_t n = 1024*100;

  double *a = new double[n];
  double *b = new double[n];
  double *c = new double[n];

  // initialize
  for(size_t i = 0; i < n; i++) {
    b[i] = (double)i;
    c[i] = 2.0*(double)i;
  }

  #pragma omp target teams loop map(from:a[:n]) map(to:b[:n],c[:n])
  for(size_t i = 0; i < n; i++) {
    a[i] = b[i] + c[i];
  }

  int err = 0;
  for(size_t i = 0; i < n; i++)
    if (a[i] != b[i]+c[i]) {
      printf("Error at %zu: got %lf, expected %lf\n", i, a[i], b[i]+c[i]);
      if (err > 10) return err;
    }

  if (!err)
    printf("Success\n");

  return err;
}
