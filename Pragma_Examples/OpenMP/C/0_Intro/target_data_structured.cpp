#include <stdio.h>
void zeros(float* a, int n) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++)
        a[i] = 0.0f;
}

void saxpy(float a, float* y, float* x, int n) {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
}

void compute_kernel_1(float *x, float *y, int n){
}
void compute_kernel_2(float *x, float *y, int n){
}

void example() {
    int N=100000;
    float tmp[N], a[N], b[N], c[N];

    #pragma omp target data map(alloc:tmp[:N])  \
                            map(to:a[:N],b[:N]) \
                            map(tofrom:c[:N])
    {
        zeros(tmp, N);
        compute_kernel_1(tmp, a, N); // uses target
        saxpy(2.0f, tmp, b, N);
        compute_kernel_2(tmp, b, N); // uses target
        saxpy(2.0f, c, tmp, N);
    }
    printf("Example program for structured target data region completed successfully\n");
}

int main(int argc, char *argv[]){
   example();
}
