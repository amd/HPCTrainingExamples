// zip_sort.hip  —  Example 1: perfect-hash / counting sort of records by 5-digit ZIP
// Build: hipcc -O3 --offload-arch=gfx942 zip_sort.hip -o zip_sort
// Run:   ./zip_sort [n]
#include <cstring>   // global ::memset visible before rocprim headers
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
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
    HIP_CHECK(hipMalloc(&zip,     (size_t)n*sizeof(int)));
    HIP_CHECK(hipMalloc(&counts,  NBUCKETS*sizeof(int)));
    HIP_CHECK(hipMalloc(&offsets, NBUCKETS*sizeof(int)));
    HIP_CHECK(hipMalloc(&out_id,  (size_t)n*sizeof(int)));

    srand(12345);
    for (int i = 0; i < n; ++i) zip[i] = rand() % NBUCKETS;
    for (int i = 0; i < NBUCKETS; ++i) counts[i] = 0;

    int dev = 0; HIP_CHECK(hipGetDevice(&dev));
    (void)hipMemPrefetchAsync(zip, (size_t)n*sizeof(int), dev);   // best-effort locality hint

    int T = 256, B = (n + T - 1)/T;

    // Warm-up: run the histogram once so the timed sort below (and the library
    // baseline) both exclude one-time GPU/context initialization -> fair compare.
    histogram<<<B,T>>>(zip, n, counts);
    HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < NBUCKETS; ++i) counts[i] = 0;   // reset for the timed run

    // Time only the sort itself (histogram + scan + scatter), not setup/allocation.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    histogram<<<B,T>>>(zip, n, counts);
    HIP_CHECK(hipDeviceSynchronize());

    thrust::exclusive_scan(thrust::device, counts, counts + NBUCKETS, offsets);
    HIP_CHECK(hipDeviceSynchronize());

    scatter<<<B,T>>>(zip, n, offsets, out_id);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float sort_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&sort_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    bool ok = true;
    for (int i = 1; i < n; ++i) if (zip[out_id[i-1]] > zip[out_id[i]]){ ok = false; break; }
    printf("n=%d  sorted=%s  first ZIP=%05d  last ZIP=%05d  sort_time=%.3f ms\n",
           n, ok ? "YES" : "NO", zip[out_id[0]], zip[out_id[n-1]], sort_ms);

    // --- Library baseline: general radix sort of the SAME keys, for comparison.
    // The custom counting sort above exploits the bounded ZIP range [0,100000);
    // a general-purpose radix sort makes no such assumption. Sort a COPY so the
    // custom result stays intact, and time it the same way (hipEvent).
    int *zip_in, *zip_out; void* tmp = nullptr; size_t tb = 0;
    HIP_CHECK(hipMalloc(&zip_in,  (size_t)n*sizeof(int)));
    HIP_CHECK(hipMalloc(&zip_out, (size_t)n*sizeof(int)));
    memcpy(zip_in, zip, (size_t)n*sizeof(int));
    HIP_CHECK((hipError_t)rocprim::radix_sort_keys(nullptr, tb, zip_in, zip_out, n));
    HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
    (void)rocprim::radix_sort_keys(tmp, tb, zip_in, zip_out, n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
    memcpy(zip_in, zip, (size_t)n*sizeof(int));

    hipEvent_t ls, le; HIP_CHECK(hipEventCreate(&ls)); HIP_CHECK(hipEventCreate(&le));
    HIP_CHECK(hipEventRecord(ls));
    (void)rocprim::radix_sort_keys(tmp, tb, zip_in, zip_out, n);
    HIP_CHECK(hipEventRecord(le)); HIP_CHECK(hipEventSynchronize(le));
    float lib_ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&lib_ms, ls, le));
    HIP_CHECK(hipEventDestroy(ls)); HIP_CHECK(hipEventDestroy(le));
    bool lib_ok = true;
    for (int i = 1; i < n; ++i) if (zip_out[i-1] > zip_out[i]){ lib_ok = false; break; }
    printf("  custom counting sort: %.3f ms (%.1f Mkeys/s)\n", sort_ms, n/(sort_ms*1.0e3));
    printf("  library radix  sort : %.3f ms (%.1f Mkeys/s)  sorted=%s  temp=%zuB\n",
           lib_ms, n/(lib_ms*1.0e3), lib_ok ? "YES" : "NO", tb);
    printf("  library/custom time ratio = %.2f  (<1 => library faster)\n", lib_ms/sort_ms);

    (void)hipFree(tmp); (void)hipFree(zip_in); (void)hipFree(zip_out);
    (void)hipFree(zip); (void)hipFree(counts); (void)hipFree(offsets); (void)hipFree(out_id);
    return 0;
}
