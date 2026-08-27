// hipfftmp_3d.cpp  —  distributed 3D C2C FFT with hipFFT-MP (native, MI300A)
//
// hipFFT-MP is the multi-process (MPI) distributed FFT that ships inside hipFFT
// in ROCm 7.2.4+ (<hipfft/hipfftMp.h>) — the AMD analog of cuFFTMp. Unlike
// heFFTe (a separate solver you build and link), hipFFT-MP needs only the ROCm
// stack: attach an MPI communicator to an ordinary hipFFT plan and the library
// handles the pencil/slab decomposition and inter-rank all-to-all for you.
//
// This uses the *built-in* decomposition: hipFFT chooses the data layout. Each
// rank owns a contiguous slab of the slowest (first) dimension of the global
// N x N x N cube, so the run requires N % nranks == 0.
//
// REQUIRES an MPI-enabled hipFFT (ROCm 7.2.4+) and an MPI library.
// Build (see Makefile target hipfftmp_3d):
//   hipcc -O3 --offload-arch=gfx942 hipfftmp_3d.cpp \
//     -I$MPI_PATH/include -L$MPI_PATH/lib -lmpi -lhipfft -o hipfftmp_3d
// Run:
//   mpirun -np 4 --map-by ppr:4:node ./hipfftmp_3d 512
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>   // hipLibXtDesc, hipfftXtMalloc/Memcpy/ExecDescriptor
#include <hipfft/hipfftMp.h>
#include <mpi.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  MPI_Abort(MPI_COMM_WORLD,1);} } while(0)
#define FFT_CHECK(x) do { hipfftResult r=(x); if(r!=HIPFFT_SUCCESS){ \
  fprintf(stderr,"hipFFT error %s:%d: %d\n",__FILE__,__LINE__,(int)r); \
  MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

// Deterministic per-point value from the global (x,y,z) coordinate so every rank
// fills its own slab independently and we can verify the round trip locally.
static inline double sample(long long x, long long y, long long z, int N){
    return std::sin(0.001 * (double)((x*N + y)*(long long)N + z));
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int N = (argc > 1) ? atoi(argv[1]) : 256;   // global N x N x N cube

    if (N % nranks != 0){
        if (rank == 0)
            fprintf(stderr, "N (%d) must be divisible by nranks (%d) for the "
                            "built-in slab decomposition.\n", N, nranks);
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank: bind to the node-local rank so co-located ranks land on
    // distinct devices.
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &shmcomm);
    int local_rank; MPI_Comm_rank(shmcomm, &local_rank);
    int ndev = 0; HIP_CHECK(hipGetDeviceCount(&ndev));
    if (ndev > 0) HIP_CHECK(hipSetDevice(local_rank % ndev));
    MPI_Comm_free(&shmcomm);

    using cpx = std::complex<double>;

    // Built-in decomposition: rank owns a slab of the slowest dimension.
    const int    local_nx = N / nranks;
    const long long x0     = (long long)rank * local_nx;
    const size_t local_sz  = (size_t)local_nx * N * N;

    // Fill this rank's input slab on the host from the global coordinates.
    std::vector<cpx> host_in(local_sz);
    for (int lx = 0; lx < local_nx; ++lx)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                host_in[((size_t)lx*N + y)*N + z] = cpx(sample(x0+lx, y, z, N), 0.0);

    // Plan: create, attach the MPI communicator, then initialize. The comm must
    // be attached *before* hipfftMakePlan3d so the plan is built distributed.
    hipfftHandle plan;
    FFT_CHECK(hipfftCreate(&plan));
    MPI_Comm comm = MPI_COMM_WORLD;
    FFT_CHECK(hipfftMpAttachComm(plan, HIPFFT_COMM_MPI, &comm));
    size_t workSize = 0;
    FFT_CHECK(hipfftMakePlan3d(plan, N, N, N, HIPFFT_Z2Z, &workSize));

    // Distributed device memory: hipFFT allocates a slab on each rank's GPU and
    // returns a descriptor. hipfftXtMemcpy scatters the host slab into it.
    hipLibXtDesc* desc = nullptr;
    FFT_CHECK(hipfftXtMalloc(plan, &desc, HIPFFT_XT_FORMAT_INPLACE));
    FFT_CHECK(hipfftXtMemcpy(plan, desc, host_in.data(), HIPFFT_COPY_HOST_TO_DEVICE));

    // Warm-up round trip (pays one-time plan/connection costs), excluded from timing.
    FFT_CHECK(hipfftXtExecDescriptor(plan, desc, desc, HIPFFT_FORWARD));
    FFT_CHECK(hipfftXtExecDescriptor(plan, desc, desc, HIPFFT_BACKWARD));
    HIP_CHECK(hipDeviceSynchronize());

    const int iters = (argc > 2) ? atoi(argv[2]) : 5;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int it = 0; it < iters; ++it){
        FFT_CHECK(hipfftXtExecDescriptor(plan, desc, desc, HIPFFT_FORWARD));
        FFT_CHECK(hipfftXtExecDescriptor(plan, desc, desc, HIPFFT_BACKWARD));
    }
    HIP_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double avg = (MPI_Wtime() - t0) / iters;   // average one fwd+bwd round trip

    // Copy the round-tripped slab back to the host and compare to the original.
    // The inverse is unnormalized, so divide by the global point count N^3.
    std::vector<cpx> host_back(local_sz);
    FFT_CHECK(hipfftXtMemcpy(plan, host_back.data(), desc, HIPFFT_COPY_DEVICE_TO_HOST));

    const double norm = (double)N * N * N;
    double local_err = 0.0;
    for (size_t i = 0; i < local_sz; ++i)
        local_err = std::max(local_err, std::abs(host_back[i] / norm - host_in[i]));
    double global_err = 0.0;
    MPI_Reduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("hipFFT-MP 3D  %dx%dx%d  ranks=%d  slab=%dx%dx%d  "
               "fwd+bwd=%.4f s (avg of %d)  round-trip max_err=%.3e\n",
               N, N, N, nranks, local_nx, N, N, avg, iters, global_err);

    FFT_CHECK(hipfftXtFree(desc));
    FFT_CHECK(hipfftDestroy(plan));
    MPI_Finalize();
    return 0;
}
