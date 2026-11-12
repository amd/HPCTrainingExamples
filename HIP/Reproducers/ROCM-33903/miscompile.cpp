// amdclang (ROCm 7.2.0) miscompilation reproducer.
// The upper half of the 64-bit array index is clobbered on a divergent path,
// so lanes taking the "else" branch store to p2 + (garbage << 34) -> GPU page fault.
// Taken from the GX gyrokinetic code (Phi2_summand / unmasked).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__constant__ int nx, ny, nyc, nz;

__device__ unsigned int get_id1(void) {return __umul24(blockIdx.x,blockDim.x)+threadIdx.x;}
__device__ unsigned int get_id2(void) {return __umul24(blockIdx.y,blockDim.y)+threadIdx.y;}
__device__ unsigned int get_id3(void) {return __umul24(blockIdx.z,blockDim.z)+threadIdx.z;}
__device__ unsigned int get_idxyz(unsigned int i, unsigned int j, unsigned int k) {return j + nyc*(i + nx*k);}
__device__ int get_ikx(int idx) { if (idx < nx/2+1) return idx; else return idx-nx; }
__device__ bool unmasked(int idx, int idy) {
  int ikx = get_ikx(idx);
  if ( !(idx==0 && idy==0) && idy < (ny-1)/3 + 1 && idx < nx
       && ikx < (nx-1)/3 + 1 && ikx > -(nx-1)/3 - 1)
    return true;
  else
    return false;
}

__global__ void Phi2_summand(float *p2, const float2 *phi, const float *volJac)
{
  unsigned int idy = get_id1(); unsigned int idx = get_id2(); unsigned int idz = get_id3();
  if (idy < nyc && idx < nx && idz < nz) {
    unsigned int idxyz = get_idxyz(idx, idy, idz);
    if (unmasked(idx, idy)) {
      float fac = 2.;
      if (idy==0) fac = 1.0;
      float2 p = phi[idxyz];
      p2[idxyz] = 0.5f * (p.x*p.x + p.y*p.y) * fac * volJac[idz];
    } else {
      p2[idxyz] = 0.;
    }
  }
}

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

int main()
{
  // GX nonlinear cyclone case: Nyc=40 (Ny=118?), Nx=192, Nz=24 -> use the same launch geometry
  int hnx = 192, hny = 118, hnyc = 40, hnz = 24;
  CHK(hipMemcpyToSymbol(HIP_SYMBOL(nx), &hnx, sizeof(int)));
  CHK(hipMemcpyToSymbol(HIP_SYMBOL(ny), &hny, sizeof(int)));
  CHK(hipMemcpyToSymbol(HIP_SYMBOL(nyc), &hnyc, sizeof(int)));
  CHK(hipMemcpyToSymbol(HIP_SYMBOL(nz), &hnz, sizeof(int)));
  size_t n = (size_t)hnx * hnyc * hnz;
  float *p2, *vj; float2 *phi;
  CHK(hipMalloc(&p2, n * sizeof(float)));
  CHK(hipMalloc(&phi, n * sizeof(float2)));
  CHK(hipMalloc(&vj, hnz * sizeof(float)));
  CHK(hipMemset(phi, 0, n * sizeof(float2)));
  CHK(hipMemset(vj, 0, hnz * sizeof(float)));
  CHK(hipMemset(p2, 0xff, n * sizeof(float)));
  dim3 dB(8, 8, 8), dG((hnyc + 7) / 8, (hnx + 7) / 8, (hnz + 7) / 8);
  printf("p2 = %p, launching...\n", (void*)p2); fflush(stdout);
  Phi2_summand<<<dG, dB>>>(p2, phi, vj);
  CHK(hipDeviceSynchronize());
  float* h = (float*)malloc(n * sizeof(float));
  CHK(hipMemcpy(h, p2, n * sizeof(float), hipMemcpyDeviceToHost));
  size_t bad = 0; for (size_t i = 0; i < n; i++) if (h[i] != 0.f) bad++;
  printf("%s: %zu of %zu elements not written\n", bad ? "FAIL" : "PASS", bad, n);
  return bad != 0;
}
