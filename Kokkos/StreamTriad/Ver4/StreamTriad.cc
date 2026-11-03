#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]){
   Kokkos::Timer timer;
   int nsize=80000000; int ntimes=16;
   double scalar = 3.0, time_sum = 0.0;

   Kokkos::initialize(argc, argv); {

   // initializing arrays
   Kokkos::View<double *> a( "a", nsize);
   Kokkos::View<double *> b( "b", nsize);
   Kokkos::View<double *> c( "c", nsize);

   Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
      a[i] = 1.0;
      b[i] = 2.0;
   });

   for (int k=0; k<ntimes; k++){
      timer.reset();
      // stream triad loop
      Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (int i) {
         c[i] = a[i] + scalar*b[i];
      });
      Kokkos::fence("parallel_for 2nd");
      time_sum += timer.seconds();
   }

   printf("Average runtime is %lf msecs\n", time_sum/ntimes*1000.0);

   } Kokkos::finalize();
}
