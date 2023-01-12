
void saxpy(int n, float a, float *restrict x, float *restrict y, float *restrict z)	
{
#pragma omp parallel for simd
        for (int i = 0; i < n; i++)
                z[i] = a*x[i] + y[i];
}
