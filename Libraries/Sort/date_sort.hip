// Copyright AMD 2026, MIT License
// Author: Bob Robey Bob.Robey@amd.com with AI tool help
//
// date_sort.hip  —  direct-address (calendar) sort on a bounded, gappy domain
//
// Framing: a NY-daily-newspaper archive. We have `n` scanned issues, each stamped
// with a UNIQUE publication date somewhere in the 4-year window
//
//     2008-01-01 (day 0)  ...  2011-12-31 (day 1460)          DOMAIN = 1461 days
//
// "One issue per date" gives us UNIQUENESS for free. The date maps to a bounded
// integer index with a plain subtraction (a *perfect* index function):
//
//     day(i) = (date_of_file[i]) - 2008-01-01            // 0-based, C indexing
//
// so we place each file with ONE write and NO atomics (unique => one writer/slot):
//
//     slot[day(i)] = i;                                  // scatter, contention-free
//
// The catch vs. a *fully dense* perfect hash: the domain is only partly filled
// (e.g. ~500 issues in 1461 days ~= 34%). The scatter leaves the array in sorted
// (calendar) order but with GAPS. So we add one step the dense case never needs:
//
//     stream-compact out the empty (-1) slots  ->  the sorted issue order.
//
// The fully-dense perfect hash (contiguous serials, ~100% full) is just the
// special case where the compaction removes nothing.
//
// This program:
//   * synthesizes n unique day-indices spread across DOMAIN (bucketed => unique)
//   * shuffles them (files arrive from the scanner in random order)
//   * sorts by direct-address scatter + stream compaction (timed, no atomics)
//   * compares against a general library radix sort of the same keys (timed)
//   * verifies the compacted order is strictly increasing by date
//   * with `dup` as argv[3]!=0, gives two files the same date to SHOW how a
//     violated uniqueness guarantee silently drops a record (a slot overwritten)
//
// Build: hipcc -O3 --offload-arch=gfx942 date_sort.hip -o date_sort
// Run:   ./date_sort [n] [domain_days] [dup(0|1)]
//   ./date_sort                 # 500 issues over the 1461-day calendar (~34% full)
//   ./date_sort 500 1461        # same, explicit
//   ./date_sort 1000000         # scales domain to keep ~34% density
//   ./date_sort 500 1461 1      # inject a duplicate date: uniqueness violated
#include <cstring>   // global ::memset visible before rocprim headers
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <cstdio>
#include <cstdint>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

// Calendar window: 2008-01-01 .. 2011-12-31 inclusive (2008 is a leap year).
static const uint32_t CAL_DOMAIN = 1461;               // 366+365+365+365
static const long     EPOCH_2008 = 13879;              // days from 1970-01-01 to 2008-01-01
static const double   TARGET_DENSITY = 0.342;          // ~500/1461 when scaling large n

// Keep only real (written) slots when compacting out the -1 sentinels.
struct not_empty { __host__ __device__ bool operator()(int v) const { return v >= 0; } };

// Direct-address scatter: slot[day[i]] = i. No atomics: unique dates => one writer.
__global__ void date_scatter(const uint32_t* __restrict__ day, int n,
                             int* __restrict__ slot){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) slot[day[i]] = i;                       // record i belongs at sorted pos day[i]
}

// Howard Hinnant's civil-from-days (days since 1970-01-01) -> (y,m,d), for printing.
static void civil_from_days(long z, int* y, unsigned* m, unsigned* d){
    z += 719468;
    long era = (z >= 0 ? z : z - 146096) / 146097;
    unsigned doe = (unsigned)(z - era*146097);                 // [0,146096]
    unsigned yoe = (doe - doe/1460 + doe/36524 - doe/146096)/365; // [0,399]
    long yr = (long)yoe + era*400;
    unsigned doy = doe - (365*yoe + yoe/4 - yoe/100);          // [0,365]
    unsigned mp  = (5*doy + 2)/153;                            // [0,11]
    *d = doy - (153*mp + 2)/5 + 1;                             // [1,31]
    *m = mp < 10 ? mp + 3 : mp - 9;                            // [1,12]
    *y = (int)(yr + (*m <= 2));
}

int main(int argc, char** argv){
    int      n      = (argc > 1) ? atoi(argv[1]) : 500;
    // Default to the real calendar; scale the domain for large n to keep density.
    uint32_t domain = (argc > 2) ? (uint32_t)strtoul(argv[2],nullptr,10)
                                 : (n <= (int)CAL_DOMAIN ? CAL_DOMAIN
                                    : (uint32_t)(n / TARGET_DENSITY));
    int      dup    = (argc > 3) ? atoi(argv[3]) : 0;

    if (n < 1 || (uint32_t)n > domain){
        fprintf(stderr,"need 1 <= n <= domain (n=%d, domain=%u). "
                       "Pass a larger domain: ./date_sort %d <domain>\n", n, domain, n);
        return 1;
    }

    // Unified memory (MI300A APU): host builds keys, GPU scatters, no hipMemcpy.
    uint32_t* day;  HIP_CHECK(hipMalloc(&day,  (size_t)n*sizeof(uint32_t)));
    int*      slot; HIP_CHECK(hipMalloc(&slot, (size_t)domain*sizeof(int)));
    int*      order;HIP_CHECK(hipMalloc(&order,(size_t)n*sizeof(int)));
    unsigned* nsel; HIP_CHECK(hipMalloc(&nsel, sizeof(unsigned)));

    // Unique day-indices: bucket i owns [i*stride, (i+1)*stride) so buckets can't
    // collide; pick one date within each bucket -> n distinct days spread over domain.
    uint32_t stride = domain / (uint32_t)n;            // >= 1 since n <= domain
    srand(2026);
    for (int i = 0; i < n; ++i) day[i] = (uint32_t)i*stride + (uint32_t)(rand() % (int)stride);
    // Files come off the scanner shuffled: permute the key array (still unique).
    for (int i = n-1; i > 0; --i){ int j = rand() % (i+1);
        uint32_t t = day[i]; day[i] = day[j]; day[j] = t; }
    if (dup && n >= 2) day[1] = day[0];                // violate uniqueness on purpose

    int T = 256, B = (n + T - 1)/T;

    // --- warm-up (excludes one-time GPU/context init from the timings) ---
    HIP_CHECK(hipMemset(slot, 0xFF, (size_t)domain*sizeof(int)));   // 0xFF ints => -1
    date_scatter<<<B,T>>>(day, n, slot); HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t s0,s1,c0,c1;
    HIP_CHECK(hipEventCreate(&s0)); HIP_CHECK(hipEventCreate(&s1));
    HIP_CHECK(hipEventCreate(&c0)); HIP_CHECK(hipEventCreate(&c1));

    // Reset the domain to empty, then time the two-stage direct-address sort.
    HIP_CHECK(hipMemset(slot, 0xFF, (size_t)domain*sizeof(int)));

    HIP_CHECK(hipEventRecord(s0));
    date_scatter<<<B,T>>>(day, n, slot);               // 1) scatter into calendar slots
    HIP_CHECK(hipEventRecord(s1)); HIP_CHECK(hipEventSynchronize(s1));

    // 2) compact out empty slots -> compacted 'order' holds file indices in date order
    void* tmp = nullptr; size_t tb = 0;
    HIP_CHECK((hipError_t)rocprim::select(nullptr, tb, slot, order, nsel, (size_t)domain, not_empty{}));
    HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
    (void)rocprim::select(tmp, tb, slot, order, nsel, (size_t)domain, not_empty{}); // warm-up
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(c0));
    HIP_CHECK((hipError_t)rocprim::select(tmp, tb, slot, order, nsel, (size_t)domain, not_empty{}));
    HIP_CHECK(hipEventRecord(c1)); HIP_CHECK(hipEventSynchronize(c1));

    float scat_ms=0.f, comp_ms=0.f;
    HIP_CHECK(hipEventElapsedTime(&scat_ms, s0, s1));
    HIP_CHECK(hipEventElapsedTime(&comp_ms, c0, c1));
    float total_ms = scat_ms + comp_ms;
    unsigned selected = *nsel;                          // unified memory: readable on host

    // Validate: compacted order is strictly increasing by date (unique => == n).
    bool order_ok = true;
    for (unsigned k = 1; k < selected; ++k)
        if (day[order[k]] <= day[order[k-1]]) { order_ok = false; break; }

    // Library baseline: a general radix sort makes no uniqueness/range assumption.
    uint32_t *kin,*kout; void* rtmp=nullptr; size_t rb=0;
    HIP_CHECK(hipMalloc(&kin, (size_t)n*sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&kout,(size_t)n*sizeof(uint32_t)));
    memcpy(kin, day, (size_t)n*sizeof(uint32_t));
    HIP_CHECK((hipError_t)rocprim::radix_sort_keys(nullptr, rb, kin, kout, n));
    HIP_CHECK(hipMalloc(&rtmp, rb ? rb : 1));
    (void)rocprim::radix_sort_keys(rtmp, rb, kin, kout, n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
    memcpy(kin, day, (size_t)n*sizeof(uint32_t));
    hipEvent_t r0,r1; HIP_CHECK(hipEventCreate(&r0)); HIP_CHECK(hipEventCreate(&r1));
    HIP_CHECK(hipEventRecord(r0));
    (void)rocprim::radix_sort_keys(rtmp, rb, kin, kout, n);
    HIP_CHECK(hipEventRecord(r1)); HIP_CHECK(hipEventSynchronize(r1));
    float lib_ms = 0.f; HIP_CHECK(hipEventElapsedTime(&lib_ms, r0, r1));

    printf("date_sort: n=%d issues  domain=%u days  density=%.1f%%  dup=%d\n",
           n, domain, 100.0*n/domain, dup);
    printf("  direct-address sort  : %.3f ms  (scatter %.3f + compact %.3f)  no atomics\n",
           total_ms, scat_ms, comp_ms);
    printf("  library radix sort   : %.3f ms  temp=%zuB\n", lib_ms, rb);
    printf("  library/direct time ratio = %.2f  (>1 => direct-address faster)\n",
           lib_ms/total_ms);
    if (selected == (unsigned)n && order_ok){
        printf("  result: %u issues in date order  (%u empty days compacted out)\n",
               selected, domain - selected);
        int y; unsigned m,d;                            // show the first few sorted dates
        unsigned show = selected < 3 ? selected : 3;
        for (unsigned k = 0; k < show; ++k){
            civil_from_days(EPOCH_2008 + (long)day[order[k]], &y, &m, &d);
            printf("    sorted[%u]: file #%d  ->  %04d-%02u-%02u\n", k, order[k], y, m, d);
        }
    } else {
        printf("  result: NOT a clean bijection  selected=%u (expected %d)  order_ok=%s\n",
               selected, n, order_ok ? "YES" : "NO");
        printf("  -> two files shared a date; one scatter overwrote the other, so a\n");
        printf("     record was silently dropped. This is why the index must be UNIQUE.\n");
    }

    (void)hipFree(tmp); (void)hipFree(rtmp); (void)hipFree(kin); (void)hipFree(kout);
    (void)hipFree(day); (void)hipFree(slot); (void)hipFree(order); (void)hipFree(nsel);
    HIP_CHECK(hipEventDestroy(s0)); HIP_CHECK(hipEventDestroy(s1));
    HIP_CHECK(hipEventDestroy(c0)); HIP_CHECK(hipEventDestroy(c1));
    HIP_CHECK(hipEventDestroy(r0)); HIP_CHECK(hipEventDestroy(r1));
    return 0;
}
