// hipFFT ROCm 7.2 regression reproducer:
// hipfftXtSetCallback() keeps the *address* of the caller's callback-pointer
// array instead of copying the pointers. If that array lives on the stack and
// goes out of scope before hipfftExec*, the kernel calls a garbage address.
// (cuFFT and ROCm <= 7.1.1 copy the values at hipfftXtSetCallback time.)
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)
#define CHKF(x) do { hipfftResult r = (x); if (r != HIPFFT_SUCCESS) { \
  printf("hipFFT error %d at %s:%d\n", (int)r, __FILE__, __LINE__); exit(1);} } while (0)

__device__ void scale_cb(void* out, size_t off, hipfftComplex el, void* data, void* shared)
{
  float s = ((float*)data)[0];
  ((hipfftComplex*)out)[off] = make_hipFloatComplex(s * el.x, s * el.y);
}
__device__ hipfftCallbackStoreC scale_cb_ptr = scale_cb;

static float* g_scale; // callback data pointer kept alive for the whole run

// Same pattern as many cuFFT codes: function pointer in a local variable.
__attribute__((noinline)) void set_callback(hipfftHandle plan)
{
  hipfftCallbackStoreC cb_h;
  CHK(hipMemcpyFromSymbol(&cb_h, HIP_SYMBOL(scale_cb_ptr), sizeof(cb_h)));
  CHKF(hipfftXtSetCallback(plan, (void**)&cb_h, HIPFFT_CB_ST_COMPLEX, (void**)&g_scale));
}

// Overwrite the stack area that previously held cb_h.
__attribute__((noinline)) void clobber_stack()
{
  volatile char buf[4096];
  for (int i = 0; i < (int)sizeof(buf); i++) buf[i] = 0x5a;
}

int main(int argc, char** argv)
{
  bool clobber = argc > 1 && !strcmp(argv[1], "clobber");
  const int N = 24;
  float *in; hipfftComplex* out;
  CHK(hipMalloc(&g_scale, sizeof(float)));
  float two = 2.f; CHK(hipMemcpy(g_scale, &two, sizeof(float), hipMemcpyHostToDevice));
  CHK(hipMalloc(&in, sizeof(float) * N));
  CHK(hipMalloc(&out, sizeof(hipfftComplex) * (N / 2 + 1)));
  float hin[N]; for (int i = 0; i < N; i++) hin[i] = 1.f;
  CHK(hipMemcpy(in, hin, sizeof(hin), hipMemcpyHostToDevice));

  hipfftHandle plan;
  CHKF(hipfftPlan1d(&plan, N, HIPFFT_R2C, 1));
  set_callback(plan);           // cb_h goes out of scope here
  if (clobber) clobber_stack(); // reuse that stack memory
  printf("executing (clobber=%d)...\n", clobber); fflush(stdout);
  CHKF(hipfftExecR2C(plan, in, out));
  CHK(hipDeviceSynchronize());
  hipfftComplex h; CHK(hipMemcpy(&h, out, sizeof(h), hipMemcpyDeviceToHost));
  printf("out[0] = (%g, %g), expected (48, 0) -> %s\n", h.x, h.y,
         (h.x == 48.f && h.y == 0.f) ? "PASS" : "FAIL");
  CHKF(hipfftDestroy(plan));
  return 0;
}
