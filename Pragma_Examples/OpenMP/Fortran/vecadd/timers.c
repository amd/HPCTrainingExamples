/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <stdlib.h>

#include "timers.h"

void cpu_timer_start(long long *sec, long long *usec){
   struct timeval tstart_cpu;
#ifdef _OPENMP
#pragma omp master
#endif
   {
     gettimeofday(&tstart_cpu, NULL);
     (*sec) = tstart_cpu.tv_sec;
     (*usec) = tstart_cpu.tv_usec;
   }
}

double cpu_timer_stop(long long tstart_sec, long long tstart_usec){
   struct timeval tstop_cpu, tresult;
   double result = 0.0;
#ifdef _OPENMP
#pragma omp master
#endif
   {
     gettimeofday(&tstop_cpu, NULL);
     tresult.tv_sec = tstop_cpu.tv_sec - tstart_sec;
     tresult.tv_usec = tstop_cpu.tv_usec - tstart_usec;
     result = (double)tresult.tv_sec + (double)tresult.tv_usec*1.0e-6;
   }
   return(result);
}

void cpu_timer_accumulate(long long tstart_sec, long long tstart_usec, double *taccumulate){
   struct timeval tstop_cpu, tresult;
#ifdef _OPENMP
#pragma omp master
#endif
   {
     gettimeofday(&tstop_cpu, NULL);
     tresult.tv_sec = tstop_cpu.tv_sec - tstart_sec;
     tresult.tv_usec = tstop_cpu.tv_usec - tstart_usec;
     (*taccumulate) += (double)tresult.tv_sec + (double)tresult.tv_usec*1.0e-6;
   }
}

