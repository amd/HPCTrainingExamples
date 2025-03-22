#include <stdio.h>

double some_computation(double x, int i){
  return(2.0*x);
}

void update_input_array_on_the_host(double *x, int N){
  for (int i=0; i<N; i++){
     x[i] = 1.0;
  }
}

double final_computation(double x, double y, int i){
  return(x+y);
}

int main(int argc, char *argv[]) {

  int N=100000;
  double tmp[N], input[N], res=0.0;

  for (int i=0; i<N; i++)
    input[i]=1.0;

#pragma omp target data map(alloc:tmp[:N]) map(to:input[:N]) map(from:res)
  {
#pragma omp target
#pragma omp teams distribute parallel for
    for (int i=0; i<N; i++)
      tmp[i] = some_computation(input[i], i);

    update_input_array_on_the_host(input, N);

#pragma omp target update to(input[:N])

#pragma omp target teams distribute parallel for reduction(+:res)
    for (int i=0; i<N; i++)
      res += final_computation(input[i], tmp[i], i);
  }

  printf("Target Update result is %lf\n",res);
}
