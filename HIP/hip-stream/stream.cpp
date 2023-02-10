/*
Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void 
vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 

  {
 
      int x = blockDim.x * blockIdx.x + threadIdx.x;
      int y = blockDim.y * blockIdx.y + threadIdx.y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = b[i] + c[i];
      }



  }

using namespace std;

int main() {
  
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;



  cout << "hip Device prop succeeded " << endl ;


  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i*100.0f;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(float), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(vectoradd_float, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);


  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  //hipResetDefaultAccelerator();

  return errors;
}
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)
    SCALE:      a(i) = q*b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q*c(i)

  It measures the memory system on the device.
  The implementation is in single precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Written by: Massimiliano Fatica, NVIDIA Corporation

  Further modifications by: Ben Cumming, CSCS

  Ported to HIP by: Peng Sun, AMD
*/

#include "hip/hip_runtime.h"
#define NTIMES  20

#include <string>
#include <vector>

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>

#include <sys/time.h>

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

typedef double real;

static double   avgtime[4] = {0}, maxtime[4] = {0},
        mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};


void print_help()
{
    printf(
        "Usage: stream [-s] [-n <elements>] [-b <blocksize>]\n\n"
        "  -s\n"
        "        Print results in SI units (by default IEC units are used)\n\n"
        "  -n <elements>\n"
        "        Put <elements> values in the arrays\n"
        "        (defaults to 1<<26)\n\n"
        "  -b <blocksize>\n"
        "        Use <blocksize> as the number of threads in each block\n"
        "        (defaults to 192)\n"
    );
}

void parse_options(int argc, char** argv, bool& SI, int& N, int& blockSize)
{
    // Default values
    SI = false;
    N = 1<<26;
    blockSize = 192;

    int c;

    while ((c = getopt (argc, argv, "sn:b:h")) != -1)
        switch (c)
        {
            case 's':
                SI = true;
                break;
            case 'n':
                N = std::atoi(optarg);
                break;
            case 'b':
                blockSize = std::atoi(optarg);
                break;
            case 'h':
                print_help();
                std::exit(0);
                break;
            default:
                print_help();
                std::exit(1);
        }
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */


double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


template <typename T>
__global__ void set_array(T *a,  T value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        a[idx] = value;
}

template <typename T>
__global__ void STREAM_Copy(T *a, T *b, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        b[idx] = a[idx];
}

template <typename T>
__global__ void STREAM_Scale(T *a, T *b, T scale,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        b[idx] = scale* a[idx];
}

template <typename T>
__global__ void STREAM_Add(T *a, T *b, T *c,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        c[idx] = a[idx]+b[idx];
}

template <typename T>
__global__ void STREAM_Triad(T *a, T *b, T *c, T scalar, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        c[idx] = a[idx]+scalar*b[idx];
}

int main(int argc, char** argv)
{
    real *d_a, *d_b, *d_c;
    int j,k;
    double times[4][NTIMES];
    real scalar;
    std::vector<std::string> label{"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

    // Parse arguments
    bool SI;
    int N, blockSize;
    parse_options(argc, argv, SI, N, blockSize);

    printf(" STREAM Benchmark implementation in HIP\n");
    printf(" Array size (%s precision) =%7.2f MB\n", sizeof(double)==sizeof(real)?"double":"single", double(N)*double(sizeof(real))/1.e6);

    /* Allocate memory on device */
    hipMalloc((void**)&d_a, sizeof(real)*N);
    hipMalloc((void**)&d_b, sizeof(real)*N);
    hipMalloc((void**)&d_c, sizeof(real)*N);

    /* Compute execution configuration */
    dim3 dimBlock(blockSize);
    dim3 dimGrid(N/dimBlock.x );
    if( N % dimBlock.x != 0 ) dimGrid.x+=1;

    printf(" using %d threads per block, %d blocks\n",dimBlock.x,dimGrid.x);

    if (SI)
        printf(" output in SI units (KB = 1000 B)\n");
    else
        printf(" output in IEC units (KiB = 1024 B)\n");

    /* Initialize memory on the device */
    hipLaunchKernelGGL(set_array<real>, dim3(dimGrid), dim3(dimBlock), 0, 0, d_a, 2.f, N);
    hipLaunchKernelGGL(set_array<real>, dim3(dimGrid), dim3(dimBlock), 0, 0, d_b, .5f, N);
    hipLaunchKernelGGL(set_array<real>, dim3(dimGrid), dim3(dimBlock), 0, 0, d_c, .5f, N);

    /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar=3.0f;
    for (k=0; k<NTIMES; k++)
    {
        times[0][k]= mysecond();
        hipLaunchKernelGGL(STREAM_Copy<real>, dim3(dimGrid), dim3(dimBlock), 0, 0, d_a, d_c, N);
        hipDeviceSynchronize();
        times[0][k]= mysecond() -  times[0][k];

        times[1][k]= mysecond();
        hipLaunchKernelGGL(STREAM_Scale<real>, dim3(dimGrid), dim3(dimBlock), 0, 0, d_b, d_c, scalar,  N);
        hipDeviceSynchronize();
        times[1][k]= mysecond() -  times[1][k];

        times[2][k]= mysecond();
        hipLaunchKernelGGL(STREAM_Add<real>, dim3(dimGrid), dim3(dimBlock), 0, 0, d_a, d_b, d_c,  N);
        hipDeviceSynchronize();
        times[2][k]= mysecond() -  times[2][k];

        times[3][k]= mysecond();
        hipLaunchKernelGGL(STREAM_Triad<real>, dim3(dimGrid), dim3(dimBlock), 0, 0, d_b, d_c, d_a, scalar,  N);
        hipDeviceSynchronize();
        times[3][k]= mysecond() -  times[3][k];
    }

    /*  --- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
    {
        for (j=0; j<4; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    double bytes[4] = {
        2 * sizeof(real) * (double)N,
        2 * sizeof(real) * (double)N,
        3 * sizeof(real) * (double)N,
        3 * sizeof(real) * (double)N
    };

    // Use right units
    const double G = SI ? 1.e9 : static_cast<double>(1<<30);

    printf("\nFunction      Rate %s  Avg time(s)  Min time(s)  Max time(s)\n",
           SI ? "(GB/s) " : "(GiB/s)" );
    printf("-----------------------------------------------------------------\n");
    for (j=0; j<4; j++) {
        avgtime[j] = avgtime[j]/(double)(NTIMES-1);

        printf("%s%11.4f     %11.8f  %11.8f  %11.8f\n", label[j].c_str(),
                bytes[j]/mintime[j] / G,
                avgtime[j],
                mintime[j],
                maxtime[j]);
    }


    /* Free memory on device */
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}

