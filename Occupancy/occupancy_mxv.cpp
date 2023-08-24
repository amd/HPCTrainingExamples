#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>

#define index(a,i,j,ld) (a[((i) * (ld)) + (j)])

#define batch 1
#define Nm 4
#define nThreads 256
#define Nunroll 4

template <typename T>
__global__ void Mxv_naive(T *A, int lda, T *x, int ldx, T *b)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  for(int i = 0; i < lda; i++)
    for(int j = 0; j < lda; j++)
      index(b,i,idx,ldx) += index(A,i,j,lda) * index(x,j,idx,ldx);
}

template <typename T>
__global__ __launch_bounds__(nThreads,1) void Mxv_shmem(T *A, int lda, T *x, int ldx, T *v)
{
  T __shared__ shm[Nm][nThreads];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int disp = batch + idx;
  
  for(int i=0; i < lda; i++)
    shm[i][threadIdx.x] = index(x,i,idx,ldx);
  
  __syncthreads();

  for(int i = 0; i < lda; i++)
    for(int j = 0; j < lda; j++)
      index(v,i,idx,ldx) += index(A,i,j,lda) * shm[j][threadIdx.x];

}

template <typename T>
__global__ __launch_bounds__(nThreads,1) void Mxv_shmem_batched(T *A, int lda, T *x, int ldx, T *v)
{
  T __shared__ shm[Nm][nThreads*batch];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  int disp = idx * batch;
  int disp_shmem = threadIdx.x * batch;
  
  for(int i=0; i < lda; i++)
    for(int b = 0; b < batch; b++)
      {
	shm[i][disp_shmem+b] = index(x,i,disp+b,ldx);
      }
  
  __syncthreads();

  for(int b=0; b < batch; b++)
    for(int i = 0; i < lda; i++)
      for(int j = 0; j < lda; j++)
	index(v,i,disp+b,ldx) += index(A,i,j,lda) * shm[j][disp_shmem+b];

}

template <typename T, int Nu>
__global__ __launch_bounds__(nThreads,1) void Mxv_shmem_batched_unroll(T *__restrict__ A, int lda, T * __restrict__ x, int ldx, T * __restrict__ v)
{
  T __shared__ shm[Nm][nThreads*batch];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  int disp = idx * batch;
  int disp_shmem = threadIdx.x * batch;
  
  for(int i=0; i < lda; i++)
    for(int b = 0; b < batch; b++)
      {
	shm[i][disp_shmem+b] = index(x,i,disp+b,ldx);
      }
  
  __syncthreads();

  for(int b=0; b < batch; b++)
    for(int i = 0; i < lda; i++)
#pragma unroll Nu
      for(int j = 0; j < lda; j++)
	index(v,i,disp+b,ldx) += index(A,i,j,lda) * shm[j][disp_shmem+b];

}

template <typename T>  __global__ __launch_bounds__(nThreads,1) void Mxv_shmem_A(T *A, int lda, T *x, int ldx, T *v)
{
  T __shared__ shm[Nm][Nm];
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(threadIdx.x < lda)
    {
      for(int i=0; i < lda; i++)
	shm[i][threadIdx.x] = index(A,i,threadIdx.x,lda);
    }
  
  __syncthreads();

  for(int i = 0; i < lda; i++)
    for(int j = 0; j < lda; j++)
      index(v,i,idx,ldx) += shm[i][j] * index(x,j,idx,ldx);

}

template <typename T>
bool verify(T *A, int lda, T *x, int ldx, T *b, T *r_b)
{
  bool valid = true;

  for(int i = 0; i < lda; i++)
    for(int j = 0; j < ldx; j++)
      for(int k = 0; k < lda; k++)
	index(b,i,j,ldx) += index(A,i,k,lda) * index(x,k,j,ldx);

  for(int i = 0; i < lda; i++)
    for(int j = 0; j < ldx; j++)
      if(index(b,i,j,ldx) != index(r_b,i,j,ldx))
  	{
  	  valid = false;
  	  break;
  	}
	
  return valid;
}

int main(void)
{
  float *A_h, *A_d;
  float *x_h, *x_d;
  float *b_h, *b_d, *b_h_v;
  int Nv = 335544320;
  int trials = 100;

  std::chrono::high_resolution_clock::time_point t1, t2;

  A_h = (float*)calloc(Nm*Nm,sizeof(float));
  x_h = (float*)calloc(Nm*Nv,sizeof(float));
  b_h = (float*)calloc(Nm*Nv,sizeof(float));
  b_h_v = (float*)calloc(Nm*Nv,sizeof(float));

  for(int i = 0; i < Nm; i++)
    for(int j = 0; j < Nm; j++)
      index(A_h,i,j,Nm) = i+j;
  
  for(int i = 0; i < Nm; i++)
    for(int j = 0; j < Nv; j++)
      index(x_h,i,j,Nv) = 1.0f;

  hipMalloc(&A_d,sizeof(float)*Nm*Nm);
  hipMalloc(&x_d,sizeof(float)*Nm*Nv);
  hipMalloc(&b_d,sizeof(float)*Nm*Nv);

  hipMemcpy(A_d,A_h,Nm*Nm*sizeof(float),hipMemcpyHostToDevice);
  hipMemcpy(x_d,x_h,Nm*Nv*sizeof(float),hipMemcpyHostToDevice);
  hipMemset(b_d,0,Nm*Nv*sizeof(float));
  
  /* Warm-up */

  int nBlocks = (Nv/batch + (nThreads-1))/nThreads;

  Mxv_shmem_batched<float><<<nBlocks,nThreads>>>(A_d, Nm, x_d, Nv, b_d);

  hipMemcpy(b_h,b_d,Nm*Nv*sizeof(float),hipMemcpyDeviceToHost);

  bool ver = verify<float>(A_h, Nm, x_h, Nv, b_h_v, b_h);

  free(b_h_v);
  
  hipMemset(b_d,0,Nm*Nv*sizeof(float));

  /* Real runs */

  t1 = std::chrono::high_resolution_clock::now();

  for(int i=0; i < trials; i++)
    Mxv_shmem_batched<float><<<nBlocks,nThreads>>>(A_d, Nm, x_d, Nv, b_d);

  hipDeviceSynchronize();

  t2 = std::chrono::high_resolution_clock::now();

  double time_elapsed_x = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

  time_elapsed_x /= trials;

  hipMemcpy(b_h,b_d,Nm*Nv*sizeof(float),hipMemcpyDeviceToHost);

  double sizes = ( (double)(Nm*Nm) * Nv * sizeof(float)) * 1.0E-09;

  std::cout << time_elapsed_x << " Time - GFLOPS " << sizes/time_elapsed_x << std::endl;

  std::cout << "Verified: " << std::boolalpha << ver << std::endl;

  // Just to print something out
  printf("%f\n",index(b_h,2,10,Nv));

  hipFree(b_d); hipFree(A_d); hipFree(x_d);
  free(b_h); free(A_h); free(x_h);

  return 0;
}
