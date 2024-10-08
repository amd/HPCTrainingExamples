#include <stdio.h>

#pragma omp requires unified_shared_memory

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


void example() {
    int N=100000;
    float tmp[N], b[N];

    zeros(tmp, N);
    zeros(b, N);
    saxpy(2.0f, tmp, b, N);

    printf("validation output b[1]: %f\n",b[1]);
    printf("validation output b[N-1]: %f\n",b[N-1]);
}

int main(int argc, char *argv[]){
   example();
}
