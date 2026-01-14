#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

#define HIP_CHECK(call)                                   \
do{                                                      \
    hipError_t gpuErr = call;                            \
    if(hipSuccess != gpuErr){                            \
        printf("GPU API Error - %s:%d: '%s'\n",          \
         __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                         \
    }                                                    \
}while(0)


typedef struct { float4 *pos, *vel; } BodySystem;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(float4 *p, float4 *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      __shared__ float3 spos[BLOCK_SIZE];
      float4 tpos = p[tile * blockDim.x + threadIdx.x];
      spos[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);
      __syncthreads();

      #pragma unroll
      for (int j = 0; j < BLOCK_SIZE; j++) {
        float dx = spos[j].x - p[i].x;
        float dy = spos[j].y - p[i].y;
        float dz = spos[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
      __syncthreads();
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);
  
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  
  int bytes = 2*nBodies*sizeof(float4);
  float *buf = (float*)malloc(bytes);
  BodySystem p = { (float4*)buf, ((float4*)buf) + nBodies };

  randomizeBodies(buf, 8*nBodies); // Init pos / vel data

  float *d_buf;
  HIP_CHECK(hipMalloc(&d_buf, bytes));
  BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0; 

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    HIP_CHECK(hipMemcpy(d_buf, buf, bytes, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(bodyForce, nBlocks, BLOCK_SIZE, 0, 0, d_p.pos, d_p.vel, dt, nBodies);
    HIP_CHECK(hipMemcpy(buf, d_buf, bytes, hipMemcpyDeviceToHost));

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vel[i].x*dt;
      p.pos[i].y += p.vel[i].y*dt;
      p.pos[i].z += p.vel[i].z*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
  }
  double avgTime = totalTime / (double)(nIters-1); 

  printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  free(buf);
  HIP_CHECK(hipFree(d_buf));
}
