// Copyright AMD 2024, MIT License, contact Bob.Robey@amd.com
#include <stdio.h>

void compute_1(int n, float *x){
   for (int i=0; i<n; i++) {
      x[i] = 1.0f;  // use 1.0
   }
}

void compute_2(int n, float *y){
   for (int i=0; i<n; i++) {
      y[i] = 2.0f;  // use 2.0
   }
}

/**
 Total the results and verify the results
*/
void compute_3(int n, float *y){
   float total=0.0f;
   for (int i=0; i<n; i++) {
      total += y[i];
   }

   // expect the output to be the sum of (a * x[i] + y[i]) where
   // x[:] is initialized to 1.0, y[:] = 2.0

   if (total == (n*4.0f)) {
      printf("PASS results are verified as correct\n");
   } else {
      printf("FAIL results are not correct. Expected %f and received %f. \n", (n*4.0f), total);
   }
}


void saxpy(int n, float a, float * x, float * y);

int main(int argc, char* argv[])
{
   int n = 1024;     // use 1024 for our example
   float a = 2.0f;  // use 2.0f for our example
   float *x = new float[n];
   float *y = new float[n];

   // allocate the device memory
   #pragma omp target data map(to:x[0:count]) map(tofrom:y[0:count])
   {
      compute_1(n, x);
      compute_2(n, y);
      #pragma omp target update to(x[0:count]) to(y[0:count]) // update x and y on the target
      saxpy(n, a, x, y);  // compute a * x[i] + y[i] in parallel
   }
   compute_3(n, y);
}

void saxpy(int n, float a, float * x, float * y) {
   #pragma omp target teams distribute parallel for
   for (int i=0; i<n; i++){
      y[i] = a * x[i] + y[i];
   }
}
