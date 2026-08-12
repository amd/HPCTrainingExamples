// heffte_3d.cpp  —  distributed 3D C2C FFT with heFFTe (rocFFT backend, MI300A)
//
// heFFTe splits the global cube across ranks (pencil/slab decomposition), runs
// local rocFFT transforms, and performs the inter-rank transposes as MPI
// all-to-all. Global order/data layout is handled by heFFTe; you supply the
// local input/output boxes for this rank.
//
// REQUIRES a heFFTe build with the ROCm backend (see build recipe at bottom).
// Build (after heFFTe is installed under $HEFFTE_ROOT):
//   hipcc -O3 --offload-arch=gfx942 heffte_3d.cpp \
//     -I$HEFFTE_ROOT/include -I$MPI_PATH/include \
//     -L$HEFFTE_ROOT/lib -L$MPI_PATH/lib -lheffte -lmpi -lrocfft -o heffte_3d
// Run:
//   sbatch run_multinode.sbatch    (adapt to launch ./heffte_3d, see exercise doc)
#include "heffte.h"
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using backend_tag = heffte::backend::rocfft;   // transforms run on the MI300A GPU

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int N = (argc > 1) ? atoi(argv[1]) : 256;   // global N x N x N cube

    // Scope block: every heFFTe object (which dups/frees an MPI communicator in
    // its destructor) must be destroyed *before* MPI_Finalize is called.
    {
    // One GPU per rank on multi-GPU nodes.
    if (heffte::gpu::device_count() > 1)
        heffte::gpu::device_set(rank % heffte::gpu::device_count());

    // Global index space and a balanced process grid over it.
    heffte::box3d<> world({0,0,0}, {N-1, N-1, N-1});
    std::array<int,3> proc_grid = heffte::proc_setup_min_surface(world, nranks);

    // Local input/output sub-boxes for this rank (same boxes for a round trip).
    std::vector<heffte::box3d<>> in_boxes  = heffte::split_world(world, proc_grid);
    std::vector<heffte::box3d<>> out_boxes = heffte::split_world(world, proc_grid);

    // Communication path: on fabrics without fast inter-node GPU RDMA, the
    // GPU-aware all-to-all can be far slower than staging through host memory.
    // HEFFTE_GPU_AWARE=0 forces the host-staged path for comparison.
    heffte::plan_options options = heffte::default_options<backend_tag>();
    const char* ga = getenv("HEFFTE_GPU_AWARE");
    if (ga && atoi(ga) == 0) options.use_gpu_aware = false;

    heffte::fft3d<backend_tag> fft(in_boxes[rank], out_boxes[rank], MPI_COMM_WORLD, options);

    using cpx = std::complex<double>;

    // Initialize local input on the CPU, then load to the GPU (malloc + memcpy).
    std::vector<cpx> host_in(fft.size_inbox());
    for (size_t i = 0; i < host_in.size(); ++i)
        host_in[i] = cpx(std::sin(0.001*(double)(i + (size_t)rank*7919)), 0.0);

    heffte::gpu::vector<cpx> gpu_in  = heffte::gpu::transfer().load(host_in);
    heffte::gpu::vector<cpx> gpu_out(fft.size_outbox());
    heffte::fft3d<backend_tag>::buffer_container<cpx> work(fft.size_workspace());

    // Warm-up: first call pays one-time costs (rocFFT plan creation, inter-node
    // UCX connection setup). Exclude it from the timing so the reported numbers
    // reflect steady-state fwd+bwd cost.
    fft.forward (gpu_in.data(),  gpu_out.data(), work.data(), heffte::scale::none);
    fft.backward(gpu_out.data(), gpu_in.data(),  work.data(), heffte::scale::full);
    (void)hipDeviceSynchronize();

    const int iters = (argc > 2) ? atoi(argv[2]) : 5;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int it = 0; it < iters; ++it){
        fft.forward (gpu_in.data(),  gpu_out.data(), work.data(), heffte::scale::none);
        fft.backward(gpu_out.data(), gpu_in.data(),  work.data(), heffte::scale::full);
    }
    (void)hipDeviceSynchronize();          // ensure all GPU work is done (single-rank has no comm to force it)
    MPI_Barrier(MPI_COMM_WORLD);
    double avg = (MPI_Wtime() - t0) / iters;   // average one fwd+bwd round trip

    // Move the round-tripped data back to the CPU and compare to the original.
    std::vector<cpx> host_back = heffte::gpu::transfer::unload(gpu_in);
    double local_err = 0.0;
    for (size_t i = 0; i < host_back.size(); ++i)
        local_err = std::max(local_err, std::abs(host_back[i] - host_in[i]));
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("heFFTe 3D  %dx%dx%d  ranks=%d  grid=%dx%dx%d  "
               "fwd+bwd=%.4f s (avg of %d)  round-trip max_err=%.3e\n",
               N, N, N, nranks, proc_grid[0], proc_grid[1], proc_grid[2],
               avg, iters, global_err);
    } // heFFTe objects destroyed here, before MPI_Finalize

    MPI_Finalize();
    return 0;
}

/* ---------------------------------------------------------------------------
Build recipe for heFFTe with the ROCm backend (once, on the login node):

  module load rocm/7.2.4 openmpi
  git clone https://github.com/icl-utk-edu/heffte.git
  cd heffte && mkdir build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=hipcc \
        -DHeffte_ENABLE_ROCM=ON \
        -DHeffte_ENABLE_FFTW=OFF \
        -DAMDGPU_TARGETS=gfx942 \
        -DCMAKE_INSTALL_PREFIX=$HOME/heffte-rocm ..
  make -j && make install
  export HEFFTE_ROOT=$HOME/heffte-rocm

Alternative (native, no external build): ROCm 7.2.4 ships hipFFT-MP
(<hipfft/hipfftMp.h>) for multi-process distributed FFT — see exercise E5-alt.
--------------------------------------------------------------------------- */
