
extern int constants[10];

#pragma omp declare target
void compute(int cindex, double *x){
   *x = 1.0 + constants[cindex];
}
#pragma omp end declare target

