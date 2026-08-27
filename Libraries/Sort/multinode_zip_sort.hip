// Copyright AMD 2026, MIT License
// Author: Bob Robey Bob.Robey@amd.com with AI tool help
//
// multinode_zip_sort.hip  —  Example 3 reference solution (MPI + HIP, MI300A)
//
// Merge-free distributed sort by 5-digit ZIP:
//   1) assume uniform boundaries; sample first 10% -> ADJUST boundaries so each
//      rank owns ~equal record counts (assume-place-adjust, no second full read)
//   2) redistribute with NEIGHBOR-TO-NEIGHBOR exchange (no all-to-all): the input
//      is already written near its owner, so with a close guess each rank only has
//      a few boundary stragglers to ship one hop to the adjacent rank (rank-1 /
//      rank+1); ONE neighbor communication usually suffices, two at most. If the
//      guessed boundaries are bad (e.g. the 10% sample collapses onto one rank) the
//      sweep needs >2 rounds -> reset the boundaries from a full-data histogram,
//      restore the original local data, and place from scratch (uncapped sweep).
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
#include <climits>

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

// Direction a record must move relative to THIS rank's owned range [bnd[rank],bnd[rank+1]):
//   -1 -> below my range, ship to the left neighbor (rank-1)
//    0 -> in my range, keep
//   +1 -> at/above my range, ship to the right neighbor (rank+1)
static inline int dir_of(int zip, const std::vector<int>& bnd, int rank){
    if (zip <  bnd[rank])   return -1;
    if (zip >= bnd[rank+1]) return +1;
    return 0;
}

// Build balanced ascending boundaries from a global histogram gh[0..NB) so that
// each rank owns ~equal record counts (bnd[0]=0, bnd[nranks]=NB).
static void build_boundaries(const std::vector<long>& gh, int nranks, std::vector<int>& bnd){
    long total = 0; for (int b = 0; b < NB; ++b) total += gh[b];
    bnd.assign(nranks + 1, NB);
    bnd[0] = 0; bnd[nranks] = NB;
    long per = (total + nranks - 1) / nranks, run = 0; int r = 1;
    for (int b = 0; b < NB && r < nranks; ++b){
        run += gh[b];
        if (run >= per){ bnd[r++] = b + 1; run = 0; }
    }
    for (; r < nranks; ++r) bnd[r] = NB;                     // pad if sample sparse
}

// One iterative neighbor-to-neighbor redistribution. Each round each rank ships its
// out-of-range records one hop toward the owner and appends what it receives; any
// received data still out-of-range is forwarded on the next round. Returns the
// number of EXCHANGE ROUNDS performed to converge (0 = already placed, 1 = a single
// neighbor communication), or -1 if it hit max_loops rounds without converging.
static int neighbor_sweep(std::vector<int>& buf, const std::vector<int>& bnd,
                          int rank, int nranks, int max_loops){
    const int left  = (rank > 0)          ? rank - 1 : MPI_PROC_NULL;
    const int right = (rank < nranks - 1) ? rank + 1 : MPI_PROC_NULL;
    for (int loop = 0; ; ++loop){
        std::vector<int> keep, toLeft, toRight;
        keep.reserve(buf.size());
        for (int z : buf){
            int d = dir_of(z, bnd, rank);
            if      (d < 0) toLeft.push_back(z);
            else if (d > 0) toRight.push_back(z);
            else            keep.push_back(z);
        }

        // Converged globally once no rank has anything left to move.
        long moved = (long)toLeft.size() + (long)toRight.size(), gmoved = 0;
        MPI_Allreduce(&moved, &gmoved, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (gmoved == 0){ buf.swap(keep); return loop; }   // loop == rounds already done
        if (loop == max_loops) return -1;                  // budget spent, still not placed

        // Exchange counts, then data. Send right / recv from left, then send left /
        // recv from right (MPI_Neighbor_alltoallv on a 1-D Cartesian comm would be
        // the collective equivalent). MPI_PROC_NULL makes the endpoints no-ops.
        int nR = (int)toRight.size(), nL = (int)toLeft.size();
        int fromLeftN = 0, fromRightN = 0;
        MPI_Sendrecv(&nR, 1, MPI_INT, right, 11, &fromLeftN,  1, MPI_INT, left,  11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&nL, 1, MPI_INT, left,  12, &fromRightN, 1, MPI_INT, right, 12,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> fromLeft(fromLeftN), fromRight(fromRightN);
        MPI_Sendrecv(toRight.data(), nR, MPI_INT, right, 13,
                     fromLeft.data(), fromLeftN, MPI_INT, left, 13,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(toLeft.data(),  nL, MPI_INT, left, 14,
                     fromRight.data(), fromRightN, MPI_INT, right, 14,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        keep.insert(keep.end(), fromLeft.begin(),  fromLeft.end());
        keep.insert(keep.end(), fromRight.begin(), fromRight.end());
        buf.swap(keep);
    }
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

    // Data is "written to the processor that owns that ZIP portion": each rank fills
    // its own slice of the key space under the uniform assumption. Because the input
    // is already near its owner, a close boundary guess needs only ONE neighbor hop
    // (two at most). skew=1 instead writes badly-distributed input (not near its
    // owner), which forces >2 hops and exercises the reset path.
    int* zip; HIP_CHECK(hipMalloc(&zip, (size_t)n_local*sizeof(int)));
    srand(1000 + rank);
    const int w    = NB / nranks;                       // uniform-assumption width
    const int lo   = rank * w;                          // this rank's assumed range [lo,hi)
    const int hi   = (rank == nranks - 1) ? NB : lo + w;
    const int span = hi - lo;
    for (long i = 0; i < n_local; ++i){
        if (skew){
            zip[i] = (rand() % 100 < 80) ? (rand() % 10000)   // 80% in the low ZIP band
                                         : (rand() % NB);      // (not pre-partitioned)
        } else if (nranks > 1 && rand() % 100 < 5){
            // ~5% "boundary spill": written just across into an ADJACENT rank so a
            // single neighbor exchange repositions it once boundaries are adjusted.
            if ((rand() & 1) && lo > 0) zip[i] = lo - 1 - (rand() % w);   // -> left  neighbor
            else if (hi < NB)           zip[i] = hi     + (rand() % w);   // -> right neighbor
            else                        zip[i] = lo     + (rand() % span);
            if (zip[i] < 0)   zip[i] = 0;
            if (zip[i] >= NB) zip[i] = NB - 1;
        } else {
            zip[i] = lo + (rand() % span);               // squarely on its owner
        }
    }

    // ---- warm up the GPU (first-touch allocation + kernel/thrust setup) so the
    //      timed local sort reflects steady-state, not cold-start, cost ----
    {
        const int wn = 4096;
        int *wz, *wc, *wo, *wout;
        HIP_CHECK(hipMalloc(&wz,   wn*sizeof(int)));
        HIP_CHECK(hipMalloc(&wc,   NB*sizeof(int)));
        HIP_CHECK(hipMalloc(&wo,   NB*sizeof(int)));
        HIP_CHECK(hipMalloc(&wout, wn*sizeof(int)));
        for (int i = 0; i < wn; ++i) wz[i] = i % NB;
        for (int b = 0; b < NB; ++b) wc[b] = 0;
        histogram<<<(wn+255)/256,256>>>(wz, wn, wc);
        HIP_CHECK(hipDeviceSynchronize());
        thrust::exclusive_scan(thrust::device, wc, wc + NB, wo);
        HIP_CHECK(hipDeviceSynchronize());
        scatter_vals<<<(wn+255)/256,256>>>(wz, wn, wo, wout);
        HIP_CHECK(hipDeviceSynchronize());
        (void)hipFree(wz); (void)hipFree(wc); (void)hipFree(wo); (void)hipFree(wout);
    }

    // ---- timers (per rank, MPI_Wtime). Barrier aligns the start so the
    //      distributed-sort total is comparable across ranks. ----
    MPI_Barrier(MPI_COMM_WORLD);
    double t_dist0 = MPI_Wtime();

    // ---- 1) assume-place-adjust boundaries from a 10% sample ----
    double t_s0 = MPI_Wtime();
    long sample = n_local / 10; if (sample < 1) sample = n_local;
    std::vector<long> sh(NB, 0);
    for (long i = 0; i < sample; ++i) sh[zip[i]]++;          // 10% local histogram
    std::vector<long> gh(NB, 0);
    MPI_Allreduce(sh.data(), gh.data(), NB, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::vector<int> bnd;
    build_boundaries(gh, nranks, bnd);          // balanced from the 10% sample
    double t_sample = MPI_Wtime() - t_s0;       // sample + adjust time (this rank)

    // ---- 2) redistribute with a neighbor-to-neighbor sweep (no all-to-all) ----
    // Keep the original local records so a reset can re-place from scratch.
    std::vector<int> orig(zip, zip + n_local);
    double t_exchange = 0.0, t_rebalance = 0.0;  // exchange vs reset-rebalance cost
    int    wasted_rounds = 0;                    // rounds thrown away before a reset

    std::vector<int> buf = orig;
    double t_e0 = MPI_Wtime();
    int loops = neighbor_sweep(buf, bnd, rank, nranks, /*max_loops=*/2);
    t_exchange += MPI_Wtime() - t_e0;
    int did_reset = 0;
    if (loops < 0){
        // Guessed boundaries needed more than two neighbor rounds: RESET boundaries
        // from a FULL-data histogram and place the data from scratch. The post-reset
        // sweep is uncapped so it always terminates (<= nranks-1 hops).
        did_reset = 1;
        wasted_rounds = 2;                       // the two budgeted rounds are discarded
        double t_r0 = MPI_Wtime();
        std::vector<long> fh(NB, 0);
        for (long i = 0; i < n_local; ++i) fh[zip[i]]++;      // full local histogram
        std::vector<long> gh_full(NB, 0);
        MPI_Allreduce(fh.data(), gh_full.data(), NB, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        build_boundaries(gh_full, nranks, bnd);
        t_rebalance = MPI_Wtime() - t_r0;        // reset overhead (this rank)
        buf = orig;
        double t_e1 = MPI_Wtime();
        loops = neighbor_sweep(buf, bnd, rank, nranks, /*max_loops=*/INT_MAX);
        t_exchange += MPI_Wtime() - t_e1;
    }

    long n_recv = (long)buf.size();
    double t_c0 = MPI_Wtime();
    int* recvbuf; HIP_CHECK(hipMalloc(&recvbuf, (size_t)(n_recv>0?n_recv:1)*sizeof(int)));
    if (n_recv > 0) memcpy(recvbuf, buf.data(), (size_t)n_recv*sizeof(int)); // APU: host-accessible
    t_exchange += MPI_Wtime() - t_c0;            // staging into the sort buffer

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
    double gt_sample, gt_ex, gt_rb, gt_local, gt_dist;
    MPI_Reduce(&t_sample,    &gt_sample, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_exchange,  &gt_ex,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_rebalance, &gt_rb,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_local,     &gt_local,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_dist,      &gt_dist,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // neighbor-exchange convergence: max loops across ranks, and whether any reset
    int gloops = 0, greset = 0, gwasted = 0;
    MPI_Reduce(&loops,         &gloops,  1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&did_reset,     &greset,  1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wasted_rounds, &gwasted, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("[rank %d] owns ZIP [%d,%d)  recv=%ld  global_offset=%ld  local_sorted=%s seam=%s"
           "  loops=%d reset=%d  |  sample=%.3f  exchange=%.3f  rebalance=%.3f  local_sort=%.3f  dist_total=%.3f ms\n",
           rank, bnd[rank], bnd[rank+1], n_recv, gofs,
           local_ok?"YES":"NO", seam_ok?"YES":"NO", loops, did_reset,
           t_sample*1e3, t_exchange*1e3, t_rebalance*1e3, t_local*1e3, t_dist*1e3);
    if (rank == 0){
        double bal = (gsum>0) ? (double)gmax / ((double)gsum/nranks) : 1.0;
        printf("GLOBAL: total=%ld  imbalance(max/avg)=%.2f  local_sorted=%s  seams=%s  neighbor_loops=%d reset=%s wasted_rounds=%d  (skew=%d)\n",
               gsum, bal, g_local?"ALL-OK":"FAIL", g_seam?"ALL-OK":"FAIL",
               gloops, greset?"YES":"NO", gwasted, skew);
        printf("GLOBAL: max times (ms): sample=%.3f  exchange=%.3f  rebalance=%.3f  local_sort=%.3f  dist_total=%.3f\n",
               gt_sample*1e3, gt_ex*1e3, gt_rb*1e3, gt_local*1e3, gt_dist*1e3);
    }

    (void)hipFree(zip); (void)hipFree(recvbuf); (void)hipFree(counts);
    (void)hipFree(offsets); (void)hipFree(out);
    MPI_Finalize();
    return 0;
}
