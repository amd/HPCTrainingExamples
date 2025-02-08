#include <iostream>
#include <omp.h>
#pragma omp requires unified_shared_memory

constexpr int N       = (1024*1024);

int main()
{
  double *A = new (std::align_val_t(128)) double[N];
  double *B = new (std::align_val_t(128)) double[N];
  double *C = new (std::align_val_t(128)) double[N];

  // initialize the data in the
  #pragma omp target teams distribute parallel for
  for (auto i=0; i<N; i++)
  {
    A[i] = 1.1;
    B[i] = i / 1.3;
  }

  //add vector elements
  #pragma omp target teams distribute parallel for
  for (int i=0; i<N; i++)
  {
        C[i] = A[i] + B[i];
  }

  // verify the results
  int errors = 0;
  for (int i = 0; i < N; i++)
  {
    if (C[i] != (A[i] + B[i]))
    {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }
  delete[] A;
  delete[] B;
  delete[] C;
}