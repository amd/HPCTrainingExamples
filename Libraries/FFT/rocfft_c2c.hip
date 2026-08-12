// rocfft_c2c.hip  —  1D batched complex-to-complex FFT with rocFFT (MI300A)
// Forward then inverse, verify round-trip error. Unified memory (no hipMemcpy).
// Build: hipcc -O3 --offload-arch=gfx942 rocfft_c2c.hip -lrocfft -o rocfft_c2c
// Run:   ./rocfft_c2c [N] [batch]
#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)
#define ROC_CHECK(x) do { rocfft_status s=(x); if(s!=rocfft_status_success){ \
  fprintf(stderr,"rocFFT error %s:%d: %d\n",__FILE__,__LINE__,(int)s); exit(1);} } while(0)

int main(int argc, char** argv){
    size_t N     = (argc > 1) ? (size_t)atol(argv[1]) : (1u<<20);
    size_t batch = (argc > 2) ? (size_t)atol(argv[2]) : 1;
    size_t total = N * batch;

    ROC_CHECK(rocfft_setup());

    // Unified memory: CPU initializes the SAME buffer the GPU transforms.
    double2 *data, *orig;
    HIP_CHECK(hipMallocManaged(&data, total*sizeof(double2)));
    HIP_CHECK(hipMallocManaged(&orig, total*sizeof(double2)));
    for (size_t b = 0; b < batch; ++b)
        for (size_t i = 0; i < N; ++i){
            double v = sin(2.0*M_PI*3.0*i/N) + 0.5*cos(2.0*M_PI*7.0*i/N);
            data[b*N+i] = double2{v, 0.0};
            orig[b*N+i] = data[b*N+i];
        }

    // Forward and inverse in-place plans.
    rocfft_plan fwd = nullptr, inv = nullptr;
    size_t len = N;
    ROC_CHECK(rocfft_plan_create(&fwd, rocfft_placement_inplace,
              rocfft_transform_type_complex_forward, rocfft_precision_double,
              1, &len, batch, nullptr));
    ROC_CHECK(rocfft_plan_create(&inv, rocfft_placement_inplace,
              rocfft_transform_type_complex_inverse, rocfft_precision_double,
              1, &len, batch, nullptr));

    // Work buffer sized to the larger plan.
    rocfft_execution_info info = nullptr;
    ROC_CHECK(rocfft_execution_info_create(&info));
    size_t wb_f=0, wb_i=0;
    ROC_CHECK(rocfft_plan_get_work_buffer_size(fwd, &wb_f));
    ROC_CHECK(rocfft_plan_get_work_buffer_size(inv, &wb_i));
    size_t wb = wb_f > wb_i ? wb_f : wb_i;
    void* wbuf = nullptr;
    if (wb){ HIP_CHECK(hipMalloc(&wbuf, wb));
             ROC_CHECK(rocfft_execution_info_set_work_buffer(info, wbuf, wb)); }

    void* ibuf[1] = { data };
    ROC_CHECK(rocfft_execute(fwd, ibuf, nullptr, info));
    HIP_CHECK(hipDeviceSynchronize());
    ROC_CHECK(rocfft_execute(inv, ibuf, nullptr, info));
    HIP_CHECK(hipDeviceSynchronize());

    // rocFFT inverse is unnormalized: divide by N, then compare to original.
    double max_err = 0.0;
    for (size_t i = 0; i < total; ++i){
        double re = data[i].x / (double)N, im = data[i].y / (double)N;
        double d  = hypot(re - orig[i].x, im - orig[i].y);
        if (d > max_err) max_err = d;
    }
    printf("rocFFT C2C  N=%zu batch=%zu  round-trip max_err=%.3e  work_buf=%zu B\n",
           N, batch, max_err, wb);

    if (wbuf) (void)hipFree(wbuf);
    (void)hipFree(data); (void)hipFree(orig);
    ROC_CHECK(rocfft_execution_info_destroy(info));
    ROC_CHECK(rocfft_plan_destroy(fwd));
    ROC_CHECK(rocfft_plan_destroy(inv));
    ROC_CHECK(rocfft_cleanup());
    return 0;
}
