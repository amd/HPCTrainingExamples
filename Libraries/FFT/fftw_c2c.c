// fftw_c2c.c  —  1D batched C2C FFT written in the *plain FFTW3 API*, run on the
// GPU by hipFFTW (MI300A).
//
// This is the same 1D transform as rocfft_c2c.hip / hipfft_c2c.hip, but the
// source contains no HIP, no device pointers, no rocFFT/hipFFT calls — it is an
// ordinary FFTW3 program. hipFFTW exports the FFTW3 symbols (fftw_*), so a
// legacy CPU FFTW3 code runs on an AMD GPU with essentially no source changes:
// swap the include and the link flag.
//
//   CPU (FFTW3):      #include <fftw3.h>            ... -lfftw3
//   GPU (hipFFTW):    #include <hipfft/hipfftw.h>   ... -lhipfftw
//
// Unlike hipFFT, hipFFTW does *not* require its buffers to be GPU-visible: it
// stages host<->device under the hood. On an APU (unified HBM) that staging is
// cheap, so the relink alone buys GPU acceleration.
//
// Compiler note: the ROCm 7.2.4 hipFFTW header pulls in C++ standard headers
// (<cstddef>, <cstdlib>), so the translation unit that includes it must be
// compiled as C++ — with hipcc use `-x c++` (or name the file .cpp). The FFTW3
// source itself is unchanged.
//
// Build: hipcc -O3 --offload-arch=gfx942 -x c++ fftw_c2c.c -lhipfftw -o fftw_c2c
// Run:   ./fftw_c2c [N] [batch]
#include <hipfft/hipfftw.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv){
    int N     = (argc > 1) ? atoi(argv[1]) : (1<<20);
    int batch = (argc > 2) ? atoi(argv[2]) : 1;
    size_t total = (size_t)N * batch;

    // FFTW-style host buffers (at least 64-bit aligned). hipFFTW handles the GPU.
    fftw_complex* data = fftw_alloc_complex(total);
    fftw_complex* orig = fftw_alloc_complex(total);
    if (!data || !orig){ fprintf(stderr, "fftw_alloc_complex failed\n"); return 1; }

    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < N; ++i){
            double v = sin(2.0*M_PI*3.0*i/N) + 0.5*cos(2.0*M_PI*7.0*i/N);
            size_t idx = (size_t)b*N + i;
            data[idx][0] = v;   data[idx][1] = 0.0;   // interleaved re, im
            orig[idx][0] = v;   orig[idx][1] = 0.0;
        }

    // Plan once on the first sequence, then reuse across the batch with the
    // "new-array" execute functions (the shipping hipFFTW supports the basic
    // plans plus fftw_execute_dft, which is all this needs).
    fftw_plan fwd = fftw_plan_dft_1d(N, data, data, FFTW_FORWARD,  FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_1d(N, data, data, FFTW_BACKWARD, FFTW_ESTIMATE);
    if (!fwd || !bwd){ fprintf(stderr, "plan creation failed\n"); return 1; }

    for (int b = 0; b < batch; ++b)
        fftw_execute_dft(fwd, data + (size_t)b*N, data + (size_t)b*N);
    for (int b = 0; b < batch; ++b)
        fftw_execute_dft(bwd, data + (size_t)b*N, data + (size_t)b*N);

    // FFTW/hipFFTW inverse is unnormalized: divide by the transform length N.
    double max_err = 0.0;
    for (size_t i = 0; i < total; ++i){
        double re = data[i][0]/(double)N, im = data[i][1]/(double)N;
        double d  = hypot(re - orig[i][0], im - orig[i][1]);
        if (d > max_err) max_err = d;
    }
    printf("hipFFTW C2C  N=%d batch=%d  round-trip max_err=%.3e\n", N, batch, max_err);

    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);
    fftw_free(data);
    fftw_free(orig);
    return 0;
}
