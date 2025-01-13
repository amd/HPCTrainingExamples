#pragma omp requires unified_shared_memory

#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char *argv[]) {
   int M=100000;
   double sum=0.0;

   double* in_h = new (align_val_t(64) ) double[M];
   double* out_h = new (align_val_t(64) ) double[M];

   for (int i=0; i<M; i++) // initialize
      in_h[i] = 1.0;

#pragma omp target teams distribute parallel for map(to:in_h) map(from:out_h)
   for (int i=0; i<M; i++){
      out_h[i] = in_h[i] * 2.0;
   }

#pragma omp target teams distribute parallel for reduction(+:sum) map(to:out_h)
   for (int i=0; i<M; i++)
     sum += out_h[i];

   cout << "Result is " << fixed << setprecision(6) << sum << endl;

   delete [] in_h;
   delete [] out_h;
}
