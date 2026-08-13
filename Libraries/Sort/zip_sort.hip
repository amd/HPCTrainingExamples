// zip_sort.hip  —  Example 1: perfect-hash / counting sort of records by 5-digit ZIP
// Build: hipcc -O3 --offload-arch=gfx942 zip_sort.hip -o zip_sort
// Run:   ./zip_sort [n]
#include <hip/hip_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

static const int NBUCKETS = 100000;   // 00000..99999

__global__ void histogram(const int* __restrict__ zip, int n, int* __restrict__ counts){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[zip[i]], 1);
}

__global__ void scatter(const int* __restrict__ zip, int n,
                        int* __restrict__ offsets, int* __restrict__ out_id){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        int pos = atomicAdd(&offsets[zip[i]], 1);
        out_id[pos] = i;
    }
}

int main(int argc, char** argv){
    int n = (argc > 1) ? atoi(argv[1]) : (1<<20);

    // --- APU EXTRA CREDIT: unified memory, no hipMemcpy anywhere ---
    int *zip, *counts, *offsets, *out_id;
    HIP_CHECK(hipMallocManaged(&zip,     (size_t)n*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&counts,  NBUCKETS*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&offsets, NBUCKETS*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&out_id,  (size_t)n*sizeof(int)));

    srand(12345);
    for (int i = 0; i < n; ++i) zip[i] = rand() % NBUCKETS;
    for (int i = 0; i < NBUCKETS; ++i) counts[i] = 0;

    int dev = 0; HIP_CHECK(hipGetDevice(&dev));
    (void)hipMemPrefetchAsync(zip, (size_t)n*sizeof(int), dev);   // best-effort locality hint

    int T = 256, B = (n + T - 1)/T;
    histogram<<<B,T>>>(zip, n, counts);
    HIP_CHECK(hipDeviceSynchronize());

    thrust::exclusive_scan(thrust::device, counts, counts + NBUCKETS, offsets);
    HIP_CHECK(hipDeviceSynchronize());

    scatter<<<B,T>>>(zip, n, offsets, out_id);
    HIP_CHECK(hipDeviceSynchronize());

    bool ok = true;
    for (int i = 1; i < n; ++i) if (zip[out_id[i-1]] > zip[out_id[i]]){ ok = false; break; }
    printf("n=%d  sorted=%s  first ZIP=%05d  last ZIP=%05d\n",
           n, ok ? "YES" : "NO", zip[out_id[0]], zip[out_id[n-1]]);

    (void)hipFree(zip); (void)hipFree(counts); (void)hipFree(offsets); (void)hipFree(out_id);
    return 0;
}
