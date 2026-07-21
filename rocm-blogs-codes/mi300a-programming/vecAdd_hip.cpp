#include <iostream>
#include <hip/hip_runtime.h>

constexpr int WIDTH   = 1024;
constexpr int HEIGHT  = 1024;
constexpr int N       = (WIDTH*HEIGHT);

constexpr int THREADS_PER_BLOCK_X = 16;
constexpr int THREADS_PER_BLOCK_Y = 16;
constexpr int THREADS_PER_BLOCK_Z = 1;

#define RS __restrict__


__global__ 
void add (
  const double* RS A, 
  const double* RS B, 
  double* RS C, 
  const int &width, const int &height
)
{
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

  int i = y * width + x;
  if ( i < (width * height)) 
  {
    C[i] = A[i] + B[i];
  }
}

int main()
{
  double *A = new (std::align_val_t(128)) double[N];
  double *B = new (std::align_val_t(128)) double[N];
  double *C = new (std::align_val_t(128)) double[N];

  // initialize the data in the
  for (auto i=0; i<N; i++)
  {
    A[i] = 1.1; 
    B[i] = i / 1.3;
  }

  //call the gpu kernel
  dim3 grid (WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y);
  dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

  add<<<grid,block,0,0>>>(A, B, C, WIDTH, HEIGHT);
  hipDeviceSynchronize();

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

  // free data allocations
  delete[] A;
  delete[] B;
  delete[] C;
}
