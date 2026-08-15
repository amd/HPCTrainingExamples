// name_sort.hip  —  Example 2 reference solution (MI300A / gfx942)
//
// Three parts on the SAME data (unified memory, no hipMemcpy):
//   A) LDS-privatized first-letter histogram   -> registration-desk balancing
//   B) alphabetical sort via thrust::sort_by_key (rocPRIM radix under the hood)
//   C) compact hash (open addressing + quadratic probing) build + verify
//      -> demonstrates handling a large, sparse key space with memory ~ #entries
//
// Build: hipcc -O3 --offload-arch=gfx942 name_sort.hip -o name_sort
// Run:   ./name_sort names.txt
#include <cstring>   // global ::memset visible before rocprim/hipcub headers
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <hipcub/hipcub.hpp>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

static const int MAXLEN   = 16;   // fixed-width storage per name
static const int KLETTERS = 13;   // letters packed into a 64-bit key (27^13 < 2^63)

// Pack up to KLETTERS letters base-27 (0 = pad => shorter prefixes sort first),
// giving a 64-bit key whose numeric order equals lexicographic order.
__host__ __device__ inline uint64_t pack_key(const char* s){
    uint64_t k = 0;
    for (int j = 0; j < KLETTERS; ++j){
        char ch = s[j];
        int v = 0;
        if      (ch >= 'a' && ch <= 'z') v = ch - 'a' + 1;
        else if (ch >= 'A' && ch <= 'Z') v = ch - 'A' + 1;
        k = k * 27 + (uint64_t)v;        // keep positional weight even for pad
    }
    return k;
}

__host__ __device__ inline int first_letter(const char* s){
    char ch = s[0];
    if (ch >= 'a' && ch <= 'z') return ch - 'a';
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    return 0;
}

__global__ void compute_keys(const char* names, int n, uint64_t* keys, int* letter){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        const char* s = names + (size_t)i*MAXLEN;
        keys[i]   = pack_key(s);
        letter[i] = first_letter(s);
    }
}

// A) LDS-privatized histogram: 26 bins live in LDS; one global atomic per bin
//    per block. Ideal use of GPU "shared memory" for a small, hot key space.
__global__ void hist_lds(const int* letter, int n, int* gcount){
    __shared__ int loc[26];
    for (int b = threadIdx.x; b < 26; b += blockDim.x) loc[b] = 0;
    __syncthreads();
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&loc[letter[i]], 1);
    __syncthreads();
    for (int b = threadIdx.x; b < 26; b += blockDim.x)
        if (loc[b]) atomicAdd(&gcount[b], loc[b]);
}

// C) Compact hash: 2-universal compression + quadratic probing (Proth-prime size)
static const uint64_t EMPTY = 0xFFFFFFFFFFFFFFFFULL;
// m is a Mersenne prime (2^31-1) so (a%m)*(code%m) stays < 2^62 (no overflow).
static const uint64_t CH_M = 2147483647ULL;
static const uint64_t CH_A = 48271ULL;      // MINSTD multiplier (< m)
static const uint64_t CH_C = 1ULL;

__device__ inline int compress(uint64_t code, int tableSize){
    uint64_t h = ((CH_A % CH_M) * (code % CH_M) + CH_C) % CH_M;
    return (int)(h % (uint64_t)tableSize);
}

__global__ void ch_insert(const uint64_t* keys, int n,
                          uint64_t* tkey, int* tcnt, int tableSize,
                          unsigned long long* probe_accum){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t key = keys[i];
    int loc0 = compress(key, tableSize);
    for (int p = 0; p < tableSize; ++p){
        long long loc = ((long long)loc0 + (long long)p*p) % tableSize;
        unsigned long long prev =
            atomicCAS((unsigned long long*)&tkey[loc], (unsigned long long)EMPTY,
                      (unsigned long long)key);
        if (prev == (unsigned long long)EMPTY || prev == (unsigned long long)key){
            atomicAdd(&tcnt[loc], 1);
            atomicAdd(probe_accum, (unsigned long long)(p + 1));
            return;
        }
    }
}

__global__ void ch_query(const uint64_t* keys, int n, const uint64_t* tkey,
                         int tableSize, int* notfound){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t key = keys[i];
    int loc0 = compress(key, tableSize);
    for (int p = 0; p < tableSize; ++p){
        long long loc = ((long long)loc0 + (long long)p*p) % tableSize;
        uint64_t k = tkey[loc];
        if (k == key)  return;
        if (k == EMPTY){ atomicAdd(notfound, 1); return; }
    }
    atomicAdd(notfound, 1);
}

static bool is_prime(long x){
    if (x < 2) return false;
    for (long d = 2; d*d <= x; ++d) if (x % d == 0) return false;
    return true;
}
static long next_prime(long x){ while(!is_prime(x)) ++x; return x; }

int main(int argc, char** argv){
    const char* path = (argc > 1) ? argv[1] : "names.txt";
    std::ifstream in(path);
    if (!in){ fprintf(stderr, "cannot open %s\n", path); return 1; }
    std::vector<std::string> names;
    for (std::string line; std::getline(in, line); )
        if (!line.empty()) names.push_back(line);
    int n = (int)names.size();
    if (n == 0){ fprintf(stderr, "no names in %s\n", path); return 1; }

    // --- unified memory: CPU fills, GPU reads, no hipMemcpy ---
    char*     nbuf;   HIP_CHECK(hipMalloc(&nbuf, (size_t)n*MAXLEN));
    uint64_t* keys;   HIP_CHECK(hipMalloc(&keys, (size_t)n*sizeof(uint64_t)));
    uint64_t* keys2;  HIP_CHECK(hipMalloc(&keys2,(size_t)n*sizeof(uint64_t)));
    int*      letter; HIP_CHECK(hipMalloc(&letter,(size_t)n*sizeof(int)));
    int*      id;     HIP_CHECK(hipMalloc(&id,   (size_t)n*sizeof(int)));
    int*      gcount; HIP_CHECK(hipMalloc(&gcount, 26*sizeof(int)));

    memset(nbuf, 0, (size_t)n*MAXLEN);
    for (int i = 0; i < n; ++i){
        strncpy(nbuf + (size_t)i*MAXLEN, names[i].c_str(), MAXLEN-1);
        id[i] = i;
    }
    for (int b = 0; b < 26; ++b) gcount[b] = 0;

    int T = 256, B = (n + T - 1)/T;
    compute_keys<<<B,T>>>(nbuf, n, keys, letter);
    HIP_CHECK(hipDeviceSynchronize());
    memcpy(keys2, keys, (size_t)n*sizeof(uint64_t));   // host copy for sorting

    // ---- Part A: desk-balancing histogram ----
    hist_lds<<<B,T>>>(letter, n, gcount);
    HIP_CHECK(hipDeviceSynchronize());
    printf("First-letter distribution (n=%d):\n", n);
    for (int b = 0; b < 26; ++b)
        if (gcount[b]) printf("  %c: %5d  (%.1f%%)\n", 'A'+b, gcount[b], 100.0*gcount[b]/n);

    // Frequency-weighted desk boundaries (4 desks with ~equal packet counts).
    const int NDESK = 4;
    printf("Suggested %d registration desks (balanced by frequency):\n", NDESK);
    { long target = (n + NDESK - 1)/NDESK, run = 0; int start = 0, desk = 0;
      for (int b = 0; b < 26; ++b){
          run += gcount[b];
          if (run >= target || b == 25){
              printf("  Desk %d: %c-%c  (%ld packets)\n", desk+1, 'A'+start, 'A'+b, run);
              run = 0; start = b+1; if (++desk == NDESK) { break; }
          }
      }
    }

    // ---- Part B: alphabetical sort (rocPRIM radix via thrust) ----
    // Time only the sort itself (setup/allocation above is excluded).
    auto t0 = std::chrono::steady_clock::now();
    thrust::sort_by_key(thrust::device, keys2, keys2 + n, id);
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    bool sorted = true;
    for (int i = 1; i < n; ++i) if (keys2[i-1] > keys2[i]){ sorted = false; break; }
    printf("Alphabetical sort: %s  first=%s  last=%s\n",
           sorted ? "OK" : "FAILED",
           nbuf + (size_t)id[0]*MAXLEN, nbuf + (size_t)id[n-1]*MAXLEN);
    printf("  sort time: %.3f ms  (%.2f Mkeys/s)\n", sort_ms, n / (sort_ms * 1.0e3));

    // ---- Part B baseline: the SAME 64-bit key sort via all three libraries ----
    // thrust::sort_by_key above dispatches to rocPRIM under the hood; here we time
    // all three the SAME way (fresh copy + one warm-up) so they are directly
    // comparable. Note: thrust allocates its own scratch INSIDE each call, so its
    // time includes that allocation; rocPRIM/hipCUB let you allocate temp once and
    // reuse it (that difference is a real reason to drop to them in hot loops).
    {
        uint64_t *kin,*kout; int *iin,*iout; void* tmp=nullptr; size_t tb=0;
        HIP_CHECK(hipMalloc(&kin,  (size_t)n*sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&kout, (size_t)n*sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&iin,  (size_t)n*sizeof(int)));
        HIP_CHECK(hipMalloc(&iout, (size_t)n*sizeof(int)));

        auto reset = [&]{ memcpy(kin, keys, (size_t)n*sizeof(uint64_t));
                          for (int i=0;i<n;++i) iin[i]=i; };
        auto elapsed = [&](hipEvent_t a, hipEvent_t b){ float m=0.f;
                          HIP_CHECK(hipEventElapsedTime(&m,a,b)); return m; };

        // thrust::sort_by_key on a fresh copy, warmed up (in-place on kin/iin)
        reset(); thrust::sort_by_key(thrust::device, kin, kin+n, iin);
        HIP_CHECK(hipDeviceSynchronize()); // warm-up
        reset();
        hipEvent_t ta,tb2; HIP_CHECK(hipEventCreate(&ta)); HIP_CHECK(hipEventCreate(&tb2));
        HIP_CHECK(hipEventRecord(ta));
        thrust::sort_by_key(thrust::device, kin, kin+n, iin);
        HIP_CHECK(hipEventRecord(tb2)); HIP_CHECK(hipEventSynchronize(tb2));
        float thr_ms = elapsed(ta,tb2);
        HIP_CHECK(hipEventDestroy(ta)); HIP_CHECK(hipEventDestroy(tb2));

        // rocPRIM radix_sort_pairs (uint64 keys, int values)
        reset();
        HIP_CHECK((hipError_t)rocprim::radix_sort_pairs(nullptr,tb,kin,kout,iin,iout,n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        (void)rocprim::radix_sort_pairs(tmp,tb,kin,kout,iin,iout,n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
        reset();
        hipEvent_t ra,rb; HIP_CHECK(hipEventCreate(&ra)); HIP_CHECK(hipEventCreate(&rb));
        HIP_CHECK(hipEventRecord(ra));
        (void)rocprim::radix_sort_pairs(tmp,tb,kin,kout,iin,iout,n);
        HIP_CHECK(hipEventRecord(rb)); HIP_CHECK(hipEventSynchronize(rb));
        float rocp_ms = elapsed(ra,rb);
        bool rocp_ok=true; for(int i=1;i<n&&rocp_ok;++i) if(kout[i-1]>kout[i]) rocp_ok=false;
        HIP_CHECK(hipEventDestroy(ra)); HIP_CHECK(hipEventDestroy(rb)); HIP_CHECK(hipFree(tmp));

        // hipCUB DeviceRadixSort::SortPairs
        tmp=nullptr; tb=0; reset();
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(nullptr,tb,kin,kout,iin,iout,n));
        HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(tmp,tb,kin,kout,iin,iout,n));
        HIP_CHECK(hipDeviceSynchronize()); // warm-up
        reset();
        hipEvent_t ha,hb; HIP_CHECK(hipEventCreate(&ha)); HIP_CHECK(hipEventCreate(&hb));
        HIP_CHECK(hipEventRecord(ha));
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(tmp,tb,kin,kout,iin,iout,n));
        HIP_CHECK(hipEventRecord(hb)); HIP_CHECK(hipEventSynchronize(hb));
        float cub_ms = elapsed(ha,hb);
        bool cub_ok=true; for(int i=1;i<n&&cub_ok;++i) if(kout[i-1]>kout[i]) cub_ok=false;
        HIP_CHECK(hipEventDestroy(ha)); HIP_CHECK(hipEventDestroy(hb)); HIP_CHECK(hipFree(tmp));

        printf("Library baseline (same keys, key-value radix sort, warmed):\n");
        printf("  thrust::sort_by_key      : %.3f ms  (%.2f Mkeys/s)  (scratch alloc'd internally)\n",
               thr_ms, n/(thr_ms*1.0e3));
        printf("  rocprim::radix_sort_pairs: %.3f ms  (%.2f Mkeys/s)  sorted=%s\n",
               rocp_ms, n/(rocp_ms*1.0e3), rocp_ok?"YES":"NO");
        printf("  hipcub::DeviceRadixSort  : %.3f ms  (%.2f Mkeys/s)  sorted=%s\n",
               cub_ms, n/(cub_ms*1.0e3), cub_ok?"YES":"NO");

        (void)hipFree(kin);(void)hipFree(kout);(void)hipFree(iin);(void)hipFree(iout);
    }

    // ---- Part C: compact hash build + verify ----
    int tableSize = (int)next_prime((long)(2*n + 1));   // load factor ~0.5
    uint64_t* tkey; HIP_CHECK(hipMalloc(&tkey, (size_t)tableSize*sizeof(uint64_t)));
    int*      tcnt; HIP_CHECK(hipMalloc(&tcnt, (size_t)tableSize*sizeof(int)));
    unsigned long long* probe; HIP_CHECK(hipMalloc(&probe, sizeof(unsigned long long)));
    int*      notfound; HIP_CHECK(hipMalloc(&notfound, sizeof(int)));
    for (int i = 0; i < tableSize; ++i){ tkey[i] = EMPTY; tcnt[i] = 0; }
    *probe = 0; *notfound = 0;

    ch_insert<<<B,T>>>(keys, n, tkey, tcnt, tableSize, probe);
    HIP_CHECK(hipDeviceSynchronize());
    ch_query<<<B,T>>>(keys, n, tkey, tableSize, notfound);
    HIP_CHECK(hipDeviceSynchronize());

    int occupied = 0;
    for (int i = 0; i < tableSize; ++i) if (tkey[i] != EMPTY) ++occupied;
    printf("Compact hash: tableSize=%d (prime)  entries=%d  distinct=%d\n",
           tableSize, n, occupied);
    printf("  load factor=%.3f  avg probes/insert=%.3f  query-miss=%d (expect 0)\n",
           (double)n/tableSize, (double)(*probe)/n, *notfound);

    (void)hipFree(nbuf); (void)hipFree(keys); (void)hipFree(keys2); (void)hipFree(letter); (void)hipFree(id);
    (void)hipFree(gcount); (void)hipFree(tkey); (void)hipFree(tcnt); (void)hipFree(probe); (void)hipFree(notfound);
    return 0;
}
