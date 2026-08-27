// hipfft_c2c.hip  —  1D batched C2C FFT with the hipFFT (cuFFT-compatible) API
// Same problem as rocfft_c2c.hip, but through the portable hipFFT interface:
// identical source compiles on AMD (rocFFT backend) and NVIDIA (cuFFT backend).
// Build: hipcc -O3 --offload-arch=gfx942 hipfft_c2c.hip -lhipfft -o hipfft_c2c
// Run:   ./hipfft_c2c [N] [batch]
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)
#define FFT_CHECK(x) do { hipfftResult r=(x); if(r!=HIPFFT_SUCCESS){ \
  fprintf(stderr,"hipFFT error %s:%d: %d\n",__FILE__,__LINE__,(int)r); exit(1);} } while(0)

int main(int argc, char** argv){
    int N     = (argc > 1) ? atoi(argv[1]) : (1<<20);
    int batch = (argc > 2) ? atoi(argv[2]) : 1;
    size_t total = (size_t)N * batch;

    hipfftDoubleComplex *data, *orig;
    HIP_CHECK(hipMallocManaged(&data, total*sizeof(hipfftDoubleComplex)));
    HIP_CHECK(hipMallocManaged(&orig, total*sizeof(hipfftDoubleComplex)));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < N; ++i){
            double v = sin(2.0*M_PI*3.0*i/N) + 0.5*cos(2.0*M_PI*7.0*i/N);
            data[(size_t)b*N+i] = hipfftDoubleComplex{v, 0.0};
            orig[(size_t)b*N+i] = data[(size_t)b*N+i];
        }

    hipfftHandle plan;
    FFT_CHECK(hipfftPlan1d(&plan, N, HIPFFT_Z2Z, batch));

    FFT_CHECK(hipfftExecZ2Z(plan, data, data, HIPFFT_FORWARD));
    HIP_CHECK(hipDeviceSynchronize());
    FFT_CHECK(hipfftExecZ2Z(plan, data, data, HIPFFT_BACKWARD));
    HIP_CHECK(hipDeviceSynchronize());

    double max_err = 0.0;
    for (size_t i = 0; i < total; ++i){
        double re = data[i].x/(double)N, im = data[i].y/(double)N;
        double d  = hypot(re - orig[i].x, im - orig[i].y);
        if (d > max_err) max_err = d;
    }
    printf("hipFFT Z2Z  N=%d batch=%d  round-trip max_err=%.3e\n", N, batch, max_err);

    FFT_CHECK(hipfftDestroy(plan));
    (void)hipFree(data); (void)hipFree(orig);
    return 0;
}
