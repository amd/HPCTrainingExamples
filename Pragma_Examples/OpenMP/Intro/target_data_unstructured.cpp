#include <stdlib.h>
#include <stdio.h>

float *tmp, *a, *b, *c;

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

void compute(int N) {
    zeros(tmp, N);
    compute_kernel_1(tmp, a, N); // uses target
    saxpy(2.0f, tmp, b, N);
    compute_kernel_2(tmp, b, N); // uses target
    saxpy(2.0f, c, tmp, N);
}

int main(int argc, char *argv[]) {
    int N = 100000;

    tmp = (float *)malloc(N*sizeof(float));
    a = (float *)malloc(N*sizeof(float));
    b = (float *)malloc(N*sizeof(float));
    c = (float *)malloc(N*sizeof(float));

    #pragma omp target enter data \
       map(alloc:tmp[:N], a[:N], b[:N], C[:N])
    compute(N);
    #pragma omp target exit data map(delete:tmp, a, b, c)
    free(tmp); free(a); free(b); free(c);

    printf("Example program for unstructured target data region completed successfully\n");
}
