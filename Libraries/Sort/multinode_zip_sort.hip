// multinode_zip_sort.hip  —  Example 3 reference solution (MPI + HIP, MI300A)
//
// Merge-free distributed sort by 5-digit ZIP:
//   1) assume uniform boundaries; sample first 10% -> ADJUST boundaries so each
//      rank owns ~equal record counts (assume-place-adjust, no second full read)
//   2) partition all local records by owner rank; MPI_Alltoallv the shuffle
//   3) local GPU counting sort of the received range
//   4) merge-free: disjoint ordered ranges => global order is the rank
//      concatenation; verify seams; MPI_Exscan gives global offsets
//
// Unified memory (APU): CPU drives MPI on the same buffers the GPU sorts.
// Build: hipcc -O3 --offload-arch=gfx942 multinode_zip_sort.hip \
//        -I$(MPI_HOME)/include -L$(MPI_HOME)/lib -lmpi -o multinode_zip_sort
#include <hip/hip_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <mpi.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

static const int NB = 100000;   // ZIP range [0,100000)

__global__ void histogram(const int* z, int n, int* c){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&c[z[i]], 1);
}
__global__ void scatter_vals(const int* z, int n, int* off, int* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){ int p = atomicAdd(&off[z[i]], 1); out[p] = z[i]; }
}

// owner rank for a ZIP given ascending boundaries bnd[0..nranks] (bnd[0]=0, bnd[nranks]=NB)
static inline int owner_of(int zip, const std::vector<int>& bnd, int nranks){
    for (int r = 0; r < nranks; ++r) if (zip >= bnd[r] && zip < bnd[r+1]) return r;
    return nranks - 1;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // ---- bind one GPU (APU) per rank using the NODE-LOCAL rank ----
    // rank%ndev only works if the global ranks are block-mapped onto nodes and
    // every rank sees all GPUs. Deriving a node-local rank from a shared-memory
    // sub-communicator is robust to any launch mapping and removes the need for
    // an external GPU-binding wrapper (e.g. setting ROCR_VISIBLE_DEVICES).
    MPI_Comm nodecomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &nodecomm);
    int local_rank; MPI_Comm_rank(nodecomm, &local_rank);
    MPI_Comm_free(&nodecomm);

    int ndev = 1; (void)hipGetDeviceCount(&ndev);
    int mydev = local_rank % ndev;
    HIP_CHECK(hipSetDevice(mydev));         // one APU per rank (node-local)
    char pcibus[64] = {0};
    (void)hipDeviceGetPCIBusId(pcibus, (int)sizeof(pcibus), mydev);
    printf("[rank %d] local_rank=%d device=%d pci=%s (ndev=%d)\n",
           rank, local_rank, mydev, pcibus, ndev);

    long n_local = (argc > 1) ? atol(argv[1]) : (1L<<22);
    int  skew    = (argc > 2) ? atoi(argv[2]) : 0;   // 1 = skewed input (demo imbalance)

    int* zip; HIP_CHECK(hipMalloc(&zip, (size_t)n_local*sizeof(int)));
    srand(1000 + rank);
    for (long i = 0; i < n_local; ++i){
        if (skew) zip[i] = (rand() % 100 < 80) ? (rand() % 10000)          // 80% in low band
                                               : (rand() % NB);
        else      zip[i] = rand() % NB;
    }

    // ---- timers (per rank, MPI_Wtime). Barrier aligns the start so the
    //      distributed-sort total is comparable across ranks. ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t_dist0 = MPI_Wtime();

    // ---- 1) assume-place-adjust boundaries from a 10% sample ----
    long sample = n_local / 10; if (sample < 1) sample = n_local;
    std::vector<long> sh(NB, 0);
    for (long i = 0; i < sample; ++i) sh[zip[i]]++;          // 10% local histogram
    std::vector<long> gh(NB, 0);
    MPI_Allreduce(sh.data(), gh.data(), NB, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    long total_sample = 0; for (int b = 0; b < NB; ++b) total_sample += gh[b];
    std::vector<int> bnd(nranks + 1);
    bnd[0] = 0; bnd[nranks] = NB;
    { long per = (total_sample + nranks - 1) / nranks, run = 0; int r = 1;
      for (int b = 0; b < NB && r < nranks; ++b){
          run += gh[b];
          if (run >= per){ bnd[r++] = b + 1; run = 0; }
      }
      for (; r < nranks; ++r) bnd[r] = NB;                   // pad if sample sparse
    }

    // ---- 2) partition ALL local records by owner; MPI_Alltoallv ----
    double t_shuf0 = MPI_Wtime();
    std::vector<int> sendcounts(nranks, 0);
    for (long i = 0; i < n_local; ++i) sendcounts[owner_of(zip[i], bnd, nranks)]++;
    std::vector<int> sdispls(nranks, 0);
    for (int r = 1; r < nranks; ++r) sdispls[r] = sdispls[r-1] + sendcounts[r-1];

    std::vector<int> sendbuf(n_local);
    std::vector<int> cursor(sdispls);
    for (long i = 0; i < n_local; ++i){
        int r = owner_of(zip[i], bnd, nranks);
        sendbuf[cursor[r]++] = zip[i];
    }

    std::vector<int> recvcounts(nranks, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> rdispls(nranks, 0);
    for (int r = 1; r < nranks; ++r) rdispls[r] = rdispls[r-1] + recvcounts[r-1];
    long n_recv = (long)rdispls[nranks-1] + recvcounts[nranks-1];

    int* recvbuf; HIP_CHECK(hipMalloc(&recvbuf, (size_t)(n_recv>0?n_recv:1)*sizeof(int)));
    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  recvbuf,        recvcounts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);
    double t_shuffle = MPI_Wtime() - t_shuf0;   // data-shuffle time (this rank)

    // ---- 3) local GPU counting sort of received records ----
    double t_ls0 = MPI_Wtime();
    int *counts, *offsets, *out;
    HIP_CHECK(hipMalloc(&counts,  NB*sizeof(int)));
    HIP_CHECK(hipMalloc(&offsets, NB*sizeof(int)));
    HIP_CHECK(hipMalloc(&out, (size_t)(n_recv>0?n_recv:1)*sizeof(int)));
    for (int b = 0; b < NB; ++b) counts[b] = 0;
    if (n_recv > 0){
        int T = 256, B = (int)((n_recv + T - 1)/T);
        histogram<<<B,T>>>(recvbuf, (int)n_recv, counts);
        HIP_CHECK(hipDeviceSynchronize());
        thrust::exclusive_scan(thrust::device, counts, counts + NB, offsets);
        HIP_CHECK(hipDeviceSynchronize());
        scatter_vals<<<B,T>>>(recvbuf, (int)n_recv, offsets, out);
        HIP_CHECK(hipDeviceSynchronize());
    }
    double t_local = MPI_Wtime() - t_ls0;       // local GPU sort time (this rank)

    bool local_ok = true;
    for (long i = 1; i < n_recv; ++i) if (out[i-1] > out[i]){ local_ok = false; break; }

    // ---- 4) merge-free: global offsets + seam check ----
    long gofs = 0;
    MPI_Exscan(&n_recv, &gofs, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) gofs = 0;

    int my_min = (n_recv > 0) ? out[0]        : NB;   // sentinel if empty
    int my_max = (n_recv > 0) ? out[n_recv-1] : -1;
    int prev_max = -1;
    if (rank > 0)        MPI_Recv(&prev_max, 1, MPI_INT, rank-1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank < nranks-1) MPI_Send(&my_max,   1, MPI_INT, rank+1, 7, MPI_COMM_WORLD);
    bool seam_ok = (rank == 0) || (n_recv == 0) || (prev_max <= my_min);
    double t_dist = MPI_Wtime() - t_dist0;      // full distributed sort (this rank)

    // per-rank balance + global validity report
    long imbalance_num = n_recv, gmax = 0, gsum = 0;
    MPI_Reduce(&imbalance_num, &gmax, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&n_recv,        &gsum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    int all_local_ok = local_ok ? 1 : 0, all_seam_ok = seam_ok ? 1 : 0, g_local, g_seam;
    MPI_Reduce(&all_local_ok, &g_local, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&all_seam_ok,  &g_seam,  1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // slowest rank per phase (max) => bounds the wall-clock of each phase
    double gt_shuf, gt_local, gt_dist;
    MPI_Reduce(&t_shuffle, &gt_shuf,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_local,   &gt_local, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_dist,    &gt_dist,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("[rank %d] owns ZIP [%d,%d)  recv=%ld  global_offset=%ld  local_sorted=%s seam=%s"
           "  |  shuffle=%.3f ms  local_sort=%.3f ms  dist_total=%.3f ms\n",
           rank, bnd[rank], bnd[rank+1], n_recv, gofs,
           local_ok?"YES":"NO", seam_ok?"YES":"NO",
           t_shuffle*1e3, t_local*1e3, t_dist*1e3);
    if (rank == 0){
        double bal = (gsum>0) ? (double)gmax / ((double)gsum/nranks) : 1.0;
        printf("GLOBAL: total=%ld  imbalance(max/avg)=%.2f  local_sorted=%s  seams=%s  (skew=%d)\n",
               gsum, bal, g_local?"ALL-OK":"FAIL", g_seam?"ALL-OK":"FAIL", skew);
        printf("GLOBAL: max times (ms): shuffle=%.3f  local_sort=%.3f  dist_total=%.3f\n",
               gt_shuf*1e3, gt_local*1e3, gt_dist*1e3);
    }

    (void)hipFree(zip); (void)hipFree(recvbuf); (void)hipFree(counts);
    (void)hipFree(offsets); (void)hipFree(out);
    MPI_Finalize();
    return 0;
}
