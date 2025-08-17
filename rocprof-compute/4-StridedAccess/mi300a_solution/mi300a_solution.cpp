#include"hip/hip_runtime.h"
#include<chrono>
#include"hipCheck.h"
#include<cmath>


__global__ 
__launch_bounds__(256) void yax(double* y,
		                double* A,
		                double* x,
		                unsigned long long n,
				unsigned long long m, 
		                double* result){
  double res = 0.0;

  for(unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x; i < n; 
	  i += gridDim.x * blockDim.x){
    double temp = 0.0;

    for(unsigned long long j = 0; j < m; j++){
      temp += A[j*n+i] * x[j];
    }
    res += y[i] * temp;
  }
  unsafeAtomicAdd(&result[0],res);
}

int main(int argc, char** argv){
  dim3 grid = dim3(2048,1,1);
  dim3 block = dim3(64,1,1);
  unsigned long long exponent = 14;
  if(argc > 1 ) exponent = atoi(argv[1]);
  unsigned long long n = 2<<exponent;
  unsigned long long m = 2<<exponent;
  
  double* y;
  double* x;
  double* A;
  double* result;
  
  hipCheck(hipMalloc(&y, n*sizeof(double)));
  hipCheck(hipMalloc(&x, m*sizeof(double)));
  hipCheck(hipMalloc(&A, n*m*sizeof(double)));
  hipCheck(hipMalloc(&result, sizeof(double)));
  
  for(unsigned long long i = 0; i < n; i++){
    y[i] = 1;
  }
  for(unsigned long long i = 0; i < m; i++){
    x[i] = 1;
  }
  for(unsigned long long i = 0; i < n*m; i++){
    A[i] = 1;
  }
  result[0] = 0.0;


  yax<<<grid,block>>>(y,A,x,n,m,result);
  hipDeviceSynchronize();
  result[0] = 0.0;
  
  auto start = std::chrono::high_resolution_clock::now();
  yax<<<grid,block>>>(y,A,x,n,m,result);
  hipDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();
  
  double expected = (double)n * (double)m;
  if(std::abs(result[0] - (double)n*(double)m) >= 0.0001) {
    printf("Answer is incorrect!\n");
    printf("result = %f, expected = %f\n",result[0],expected);
  } else {
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1e-6;
    printf("yAx time: %f milliseconds\n", time);
  }

  hipFree(y);
  hipFree(x);
  hipFree(A);

  return 0;
}

