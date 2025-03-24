#include <cstddef>
#include <new>
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

#ifndef BLOCKSIZE
#define BLOCKSIZE 1024
#endif

#pragma omp requires unified_shared_memory
int main(int argc, char *argv[]){
   double *X, *Y, *Z;
   size_t N = (size_t) BLOCKSIZE*BLOCKSIZE*BLOCKSIZE/sizeof(double);
   int niter = 100;

   int alignment_length = 16;
   high_resolution_clock::time_point t1, t2;

   printf("argc is %d %s\n",argc,argv[1]);
   if (argc > 1) {
      alignment_length = atoi(argv[1]);
   }
   if (argc > 2) {
      N = atoi(argv[2]);
   }

   if (alignment_length < 16) {
      X = (double *)malloc(N*sizeof(double));
      Y = (double *)malloc(N*sizeof(double));
   } else {
      X = new (std::align_val_t(alignment_length)) double[N];
      Y = new (std::align_val_t(alignment_length)) double[N];
   }

// warm up loops and data transfer to GPU
   #pragma omp target teams distribute parallel for thread_limit(BLOCKSIZE)
   for (size_t i = 0; i < N; ++i)
      Y[i] = X[i];

   t1 = high_resolution_clock::now();

   for (int i = 0; i < niter; i++) {
      #pragma omp target teams distribute parallel for thread_limit(BLOCKSIZE)
      for (size_t i = 0; i < N; ++i)
         Y[i] = X[i];
   }

   t2 = high_resolution_clock::now();
   auto tm_duration = duration_cast<microseconds>(t2 - t1).count();

   // one copy kernel with 2 data loads
   double GB=2.0*(double)N*8.0/1024.0/1024.0/1024.0;
   // timing in microseconds converted to secs
   double secs=tm_duration/1000.0/1000.0;
   double SecsPerIter = secs/(double)niter;
   cout << "Simple Copy Took " << tm_duration/(double)niter << " microseconds for alignment length " << alignment_length << ", thread_limit(BLOCKSIZE) " << BLOCKSIZE << ", memory loads+writes " << GB << " GiB " << endl;
   cout << "Application bandwidth using one operation per memory write   " << GB/SecsPerIter << " GiB/sec or " << GB/1024.0/SecsPerIter << " TiB/sec" << endl;
   cout << "Hardware bandwidth accounting for write needing a load+store " << 3.0*GB/2.0/SecsPerIter << " GiB/sec or " << 3.0*GB/2.0/1024.0/SecsPerIter << " TiB/sec" << endl << endl;

   delete[] X;
   delete[] Y;
   return 0;
}
