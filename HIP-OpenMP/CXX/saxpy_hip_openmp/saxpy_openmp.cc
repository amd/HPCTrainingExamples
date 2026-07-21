// Copyright AMD 2024, MIT License, contact Bob.Robey@amd.com

void saxpy_openmp(int n, float a, float * x, float * y) {
   #pragma omp target teams distribute parallel for is_device_ptr(x,y)
   for (int i=0; i<n; i++){
      y[i] = a * x[i] + y[i];
   }
}


