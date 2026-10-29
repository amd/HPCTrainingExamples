#include <stdio.h>
#include "timer.h"

int main(int argc, char *argv[]){
   struct timespec tstart;
   int nsize=80000000; int ntimes=16;
   double scalar = 3.0, time_sum = 0.0;

   // initializing arrays
   double a[nsize];
   double b[nsize];
   double c[nsize];

   for (int i=0; i<nsize; i++) {
      a[i] = 1.0;
      b[i] = 2.0;
   }

   for (int k=0; k<ntimes; k++){
      cpu_timer_start(&tstart);
      // stream triad loop
      for (int i=0; i<nsize; i++){
         c[i] = a[i] + scalar*b[i];
      }
      time_sum += cpu_timer_stop(tstart);
   }

   printf("Average runtime is %lf msecs\n", time_sum/ntimes*1000.0);
}
