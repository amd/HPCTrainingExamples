// Copyright AMD 2026, MIT License
// Author: Bob Robey Bob.Robey@amd.com with AI tool help
//
// id_sort.hip  —  a TRUE perfect-hash sort: unique, dense keys => one scatter
//
// A *perfect hash* needs two guarantees:
//
//   1) DENSE bounded key range:  key in [BASE, BASE+n)   (range ~ n)
//   2) UNIQUE keys:              each key occurs exactly once  (bijection)
//
// With both guarantees, placement is a single write per element with NO atomics, NO
// histogram, NO scan:            slot[key[i]-BASE] = i;
// Every destination slot is written exactly once, so there is no contention.
//
// Real-world framing: a mailing batch where each mailpiece carries a UNIQUE
// sequential serial number (e.g. a USPS Intelligent Mail serial). The pieces come
// back from the sorter in shuffled order; we reassemble canonical order by serial.
//
// This program:
//   * builds a shuffled permutation of [BASE, BASE+n)  (guaranteed unique+dense)
//   * reorders with the perfect-hash scatter (timed, no atomics)
//   * compares against a general library radix sort of the same keys (timed)
//   * verifies the result is an exact permutation
//   * with `dup` as argv[2]!=0, injects a duplicate key to SHOW how a violated
//     uniqueness guarantee silently loses a record (a slot left unwritten)
//
// Build: hipcc -O3 --offload-arch=gfx942 id_sort.hip -o id_sort
// Run:   ./id_sort [n] [dup(0|1)]
#include <cstring>   // global ::memset visible before rocprim headers
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <cstdio>
#include <cstdint>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

static const uint64_t KEY_BASE = 1000000000ULL;   // campaign serial base

// Perfect-hash scatter: slot[key[i]-BASE] = i. No atomics: the uniqueness +
// dense-range guarantee means each slot has exactly one writer.
__global__ void perfect_scatter(const uint64_t* __restrict__ key, int n,
                                uint64_t base, int* __restrict__ slot){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        size_t d = (size_t)(key[i] - base);   // dense index in [0,n)
        slot[d] = i;                           // record i belongs at sorted pos d
    }
}

int main(int argc, char** argv){
    int n   = (argc > 1) ? atoi(argv[1]) : (1<<20);
    int dup = (argc > 2) ? atoi(argv[2]) : 0;

    // Unified memory: host builds the shuffled keys, GPU scatters, no hipMemcpy.
    uint64_t* key; HIP_CHECK(hipMalloc(&key, (size_t)n*sizeof(uint64_t)));
    int*      slot;HIP_CHECK(hipMalloc(&slot,(size_t)n*sizeof(int)));

    // Build a permutation of [BASE, BASE+n): identity then Fisher-Yates shuffle.
    for (int i = 0; i < n; ++i) key[i] = KEY_BASE + (uint64_t)i;
    srand(2026);
    for (int i = n-1; i > 0; --i){ int j = rand() % (i+1);
        uint64_t t = key[i]; key[i] = key[j]; key[j] = t; }
    if (dup && n >= 2) key[1] = key[0];   // violate uniqueness on purpose

    for (int i = 0; i < n; ++i) slot[i] = -1;   // sentinel: detect unwritten slots

    int T = 256, B = (n + T - 1)/T;

    // Warm-up so the timed scatter excludes one-time GPU/context init.
    perfect_scatter<<<B,T>>>(key, n, KEY_BASE, slot); HIP_CHECK(hipDeviceSynchronize());
    for (int i = 0; i < n; ++i) slot[i] = -1;

    hipEvent_t s0,s1; HIP_CHECK(hipEventCreate(&s0)); HIP_CHECK(hipEventCreate(&s1));
    HIP_CHECK(hipEventRecord(s0));
    perfect_scatter<<<B,T>>>(key, n, KEY_BASE, slot);
    HIP_CHECK(hipEventRecord(s1)); HIP_CHECK(hipEventSynchronize(s1));
    float perf_ms = 0.f; HIP_CHECK(hipEventElapsedTime(&perf_ms, s0, s1));
    HIP_CHECK(hipEventDestroy(s0)); HIP_CHECK(hipEventDestroy(s1));

    // Validate: every slot written (no sentinel) AND key[slot[d]] == BASE+d.
    int unwritten = 0; bool order_ok = true;
    for (int d = 0; d < n; ++d){
        if (slot[d] < 0){ ++unwritten; continue; }
        if (key[slot[d]] != KEY_BASE + (uint64_t)d) order_ok = false;
    }

    // Library baseline: a general radix sort makes no uniqueness/range assumption.
    uint64_t *kin,*kout; void* tmp=nullptr; size_t tb=0;
    HIP_CHECK(hipMalloc(&kin, (size_t)n*sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&kout,(size_t)n*sizeof(uint64_t)));
    memcpy(kin, key, (size_t)n*sizeof(uint64_t));
    HIP_CHECK((hipError_t)rocprim::radix_sort_keys(nullptr, tb, kin, kout, n));
    HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
    (void)rocprim::radix_sort_keys(tmp, tb, kin, kout, n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
    memcpy(kin, key, (size_t)n*sizeof(uint64_t));
    hipEvent_t r0,r1; HIP_CHECK(hipEventCreate(&r0)); HIP_CHECK(hipEventCreate(&r1));
    HIP_CHECK(hipEventRecord(r0));
    (void)rocprim::radix_sort_keys(tmp, tb, kin, kout, n);
    HIP_CHECK(hipEventRecord(r1)); HIP_CHECK(hipEventSynchronize(r1));
    float lib_ms = 0.f; HIP_CHECK(hipEventElapsedTime(&lib_ms, r0, r1));
    HIP_CHECK(hipEventDestroy(r0)); HIP_CHECK(hipEventDestroy(r1));

    printf("id_sort: n=%d  key range=[%llu,%llu)  dup=%d\n",
           n, (unsigned long long)KEY_BASE, (unsigned long long)(KEY_BASE+n), dup);
    printf("  perfect-hash scatter : %.3f ms (%.1f Mkeys/s)  no atomics/scan\n",
           perf_ms, n/(perf_ms*1.0e3));
    printf("  library radix sort   : %.3f ms (%.1f Mkeys/s)  temp=%zuB\n",
           lib_ms, n/(lib_ms*1.0e3), tb);
    printf("  library/perfect time ratio = %.2f  (>1 => perfect hash faster)\n",
           lib_ms/perf_ms);
    if (unwritten == 0 && order_ok){
        printf("  result: exact permutation  (all %d slots written, order verified)\n", n);
    } else {
        printf("  result: NOT a bijection  unwritten_slots=%d  order_ok=%s\n",
               unwritten, order_ok ? "YES" : "NO");
        printf("  -> a duplicate key overwrote a slot and left another empty;\n");
        printf("     this is exactly why the perfect hash requires UNIQUE keys.\n");
    }

    (void)hipFree(tmp); (void)hipFree(kin); (void)hipFree(kout);
    (void)hipFree(key); (void)hipFree(slot);
    return 0;
}
