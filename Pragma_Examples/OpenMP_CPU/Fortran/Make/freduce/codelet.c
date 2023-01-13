
double reduction(int n, double *restrict x, double *restrict y)
{
  double sum = 0;
  #pragma omp parallel for simd reduction(+:sum)
  for(int i = 0; i < n; i++)
    sum += x[i] * y[i];

  return sum;  
}
