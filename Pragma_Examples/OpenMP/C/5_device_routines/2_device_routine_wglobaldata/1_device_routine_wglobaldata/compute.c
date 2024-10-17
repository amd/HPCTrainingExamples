
#pragma omp declare target
extern double constants[10];
#pragma omp end declare target

#pragma omp declare target
void compute(int cindex, double *x){
   *x = 1.0 + constants[cindex];
}
#pragma omp end declare target

