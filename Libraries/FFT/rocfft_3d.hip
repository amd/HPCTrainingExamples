// rocfft_3d.hip  —  3D complex-to-complex FFT with rocFFT (MI300A)
// Single-device 3D transform + inverse round-trip. Motivates the multi-node
// (heFFTe) version: a 3D FFT is three batched 1D passes with transposes; across
// nodes those transposes become all-to-all communication.
// Build: hipcc -O3 --offload-arch=gfx942 rocfft_3d.hip -lrocfft -o rocfft_3d
// Run:   ./rocfft_3d [N]     (uses an N x N x N cube)
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
    size_t N = (argc > 1) ? (size_t)atol(argv[1]) : 128;   // N^3 cube
    size_t total = N*N*N;

    ROC_CHECK(rocfft_setup());

    double2 *data, *orig;
    HIP_CHECK(hipMallocManaged(&data, total*sizeof(double2)));
    HIP_CHECK(hipMallocManaged(&orig, total*sizeof(double2)));
    for (size_t i = 0; i < total; ++i){
        double v = sin(0.001*i) + 0.25*cos(0.003*i);
        data[i] = double2{v, 0.0}; orig[i] = data[i];
    }

    rocfft_plan fwd = nullptr, inv = nullptr;
    size_t len[3] = { N, N, N };   // fastest dimension first
    ROC_CHECK(rocfft_plan_create(&fwd, rocfft_placement_inplace,
              rocfft_transform_type_complex_forward, rocfft_precision_double,
              3, len, 1, nullptr));
    ROC_CHECK(rocfft_plan_create(&inv, rocfft_placement_inplace,
              rocfft_transform_type_complex_inverse, rocfft_precision_double,
              3, len, 1, nullptr));

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

    double norm = (double)total, max_err = 0.0;
    for (size_t i = 0; i < total; ++i){
        double re = data[i].x/norm, im = data[i].y/norm;
        double d  = hypot(re - orig[i].x, im - orig[i].y);
        if (d > max_err) max_err = d;
    }
    printf("rocFFT 3D  %zux%zux%zu  round-trip max_err=%.3e  work_buf=%zu B\n",
           N, N, N, max_err, wb);

    if (wbuf) (void)hipFree(wbuf);
    (void)hipFree(data); (void)hipFree(orig);
    ROC_CHECK(rocfft_execution_info_destroy(info));
    ROC_CHECK(rocfft_plan_destroy(fwd));
    ROC_CHECK(rocfft_plan_destroy(inv));
    ROC_CHECK(rocfft_cleanup());
    return 0;
}
