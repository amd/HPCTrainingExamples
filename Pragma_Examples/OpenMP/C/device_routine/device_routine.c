// Copyright (c) 2024 AMD HPC Application Performance Team
// Author: Bob Robey, Bob.Robey@amd.com
// MIT License

#include <stdio.h>

int main(int argc, char *argv[]){
   double ce[2]={0.0, 0.0};
#pragma target teams distribute parallel do simd reduction(+:ce1[0:2])
   for (int j = 0; j< 1000; j++){
      ce[0] += 1.0;
      ce[1] += 1.0;
   }

   printf("ce[0] = %lf ce[1] = %lf\n", ce[0], ce[1]);
   return(0);
}


#include <stdlib.h>
#include <stdio.h>

void compute(double *x);

int main(int argc, char *argv[]){
   int N=1000;
   double *x = (double *)malloc(N*sizeof(double));
#pragma omp target enter data map (alloc:x[0:N])
#pragma omp target teams distribute parallel for simd
   for (int k = 0; k < N; k++){
      x[k] = 0.0;
   }

#pragma omp target teams distribute parallel for simd
   for (int k = 0; k < N; k++){
      compute(&x[k]);
   }

   double sum = 0.0;
#pragma omp target teams distribute parallel for simd reduction(+:sum)
   for (int k = 0; k < N; k++){
      sum += x[k];
   }

   printf("Result: sum of x is %lf\n",sum);

#pragma omp target exit data map (release:x[0:N])
   free(x);
}
      
