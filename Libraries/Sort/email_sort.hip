// Copyright AMD 2026, MIT License
// Author: Bob Robey Bob.Robey@amd.com with AI tool help
//
// email_sort.hip  —  Example 6: unique BUT sparse keys => compact hash
//
// Emails are the case the ZIP counting sort and the date_sort direct-address hash
// cannot handle: each recipient is UNIQUE (guarantee #2) but the key is a variable-
// length string over a huge, sparse space (fails guarantee #1, a bounded domain).
// So there is no dense array to scatter into. Two things happen instead:
//
//   1) reduce each email string to a 64-bit key with a device hash (FNV-1a here),
//   2) store the keys in a COMPACT hash table sized ~ n (not ~ key range), with
//      collisions from the size-n compression resolved by open addressing.
//
// The teaching point: even with 100% UNIQUE emails, once you compress the 64-bit
// hash down to a table of size ~ n you still get collisions, so you still have to probe.
// Uniqueness gives distinct == n (no lost records); it does NOT give you a
// collision-free perfect hash. We sweep the load factor to show probes vs. load.
//
// Build: hipcc -O3 --offload-arch=gfx942 email_sort.hip -o email_sort
// Run:   ./email_sort [emails.txt]
#include <cstring>   // global ::memset visible before rocprim headers
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

static const int MAXLEN = 48;   // fixed-width storage per email

// FNV-1a 64-bit string hash: cheap, good distribution, one multiply/xor per byte.
__host__ __device__ inline uint64_t fnv1a(const char* s, int maxlen){
    uint64_t h = 1469598103934665603ULL;               // FNV offset basis
    for (int j = 0; j < maxlen && s[j]; ++j){
        h ^= (uint64_t)(unsigned char)s[j];
        h *= 1099511628211ULL;                         // FNV prime
    }
    return h;
}
__global__ void hash_emails(const char* buf, int n, uint64_t* keys){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) keys[i] = fnv1a(buf + (size_t)i*MAXLEN, MAXLEN);
}

// Compact hash: 2-universal compression of the 64-bit key + quadratic probing.
static const uint64_t EMPTY = 0xFFFFFFFFFFFFFFFFULL;
static const uint64_t CH_M = 2147483647ULL;            // 2^31-1 prime
static const uint64_t CH_A = 48271ULL, CH_C = 1ULL;
__device__ inline int compress(uint64_t code, int tableSize){
    uint64_t h = ((CH_A % CH_M) * (code % CH_M) + CH_C) % CH_M;
    return (int)(h % (uint64_t)tableSize);
}
__global__ void ch_insert(const uint64_t* keys, int n, uint64_t* tkey, int* tcnt,
                          int tableSize, unsigned long long* probe_accum){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint64_t key = keys[i]; int loc0 = compress(key, tableSize);
    for (int p = 0; p < tableSize; ++p){
        long long loc = ((long long)loc0 + (long long)p*p) % tableSize;
        unsigned long long prev = atomicCAS((unsigned long long*)&tkey[loc],
            (unsigned long long)EMPTY, (unsigned long long)key);
        if (prev == (unsigned long long)EMPTY || prev == (unsigned long long)key){
            atomicAdd(&tcnt[loc], 1);
            atomicAdd(probe_accum, (unsigned long long)(p + 1)); return; }
    }
}
__global__ void ch_query(const uint64_t* keys, int n, const uint64_t* tkey,
                         int tableSize, int* notfound){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    uint64_t key = keys[i]; int loc0 = compress(key, tableSize);
    for (int p = 0; p < tableSize; ++p){
        long long loc = ((long long)loc0 + (long long)p*p) % tableSize;
        uint64_t k = tkey[loc];
        if (k == key)  return;
        if (k == EMPTY){ atomicAdd(notfound, 1); return; }
    }
    atomicAdd(notfound, 1);
}
static bool is_prime(long x){ if(x<2)return false;
    for(long d=2; d*d<=x; ++d) if(x%d==0) return false; return true; }
static long next_prime(long x){ while(!is_prime(x)) ++x; return x; }

int main(int argc, char** argv){
    const char* path = (argc > 1) ? argv[1] : "emails.txt";
    std::ifstream in(path);
    if (!in){ fprintf(stderr, "cannot open %s (run: make emails.txt)\n", path); return 1; }
    std::vector<std::string> emails;
    for (std::string line; std::getline(in, line); ) if(!line.empty()) emails.push_back(line);
    int n = (int)emails.size();
    if (n == 0){ fprintf(stderr, "no emails in %s\n", path); return 1; }

    // Unified memory: host fills, GPU reads, no hipMemcpy.
    char*     nbuf; HIP_CHECK(hipMalloc(&nbuf, (size_t)n*MAXLEN));
    uint64_t* keys; HIP_CHECK(hipMalloc(&keys, (size_t)n*sizeof(uint64_t)));
    memset(nbuf, 0, (size_t)n*MAXLEN);
    for (int i = 0; i < n; ++i) strncpy(nbuf + (size_t)i*MAXLEN, emails[i].c_str(), MAXLEN-1);

    int T = 256, B = (n + T - 1)/T;
    hash_emails<<<B,T>>>(nbuf, n, keys); HIP_CHECK(hipDeviceSynchronize());

    printf("email_sort: n=%d unique emails, hashed to 64-bit (FNV-1a)\n", n);
    printf("  first=%s  last=%s\n\n", emails.front().c_str(), emails.back().c_str());

    // ---- Compact hash + load-factor sweep (probes vs. load factor) ----
    long maxTable = next_prime((long)(4*(long)n + 1));   // size for the loosest factor
    uint64_t* tkey; HIP_CHECK(hipMalloc(&tkey, (size_t)maxTable*sizeof(uint64_t)));
    int*      tcnt; HIP_CHECK(hipMalloc(&tcnt, (size_t)maxTable*sizeof(int)));
    unsigned long long* probe; HIP_CHECK(hipMalloc(&probe, sizeof(unsigned long long)));
    int*      notfound; HIP_CHECK(hipMalloc(&notfound, sizeof(int)));

    const double factors[] = { 1.1, 1.3, 1.5, 2.0, 3.0, 4.0 };
    printf("Compact hash (open addressing + quadratic probing), unique keys:\n");
    printf("  %-10s %-12s %-9s %-16s %-9s %-8s\n",
           "factor", "tableSize", "load", "avg_probes/ins", "distinct", "miss");
    for (double f : factors){
        int tableSize = (int)next_prime((long)(f*(double)n + 1));
        for (int i = 0; i < tableSize; ++i){ tkey[i] = EMPTY; tcnt[i] = 0; }
        *probe = 0; *notfound = 0;
        ch_insert<<<B,T>>>(keys, n, tkey, tcnt, tableSize, probe); HIP_CHECK(hipDeviceSynchronize());
        ch_query <<<B,T>>>(keys, n, tkey, tableSize, notfound);    HIP_CHECK(hipDeviceSynchronize());
        int occupied = 0; for (int i = 0; i < tableSize; ++i) if (tkey[i] != EMPTY) ++occupied;
        printf("  %-10.2f %-12d %-9.3f %-16.3f %-9d %-8d\n",
               f, tableSize, (double)n/tableSize, (double)(*probe)/n, occupied, *notfound);
    }
    printf("\ndistinct == n confirms UNIQUE keys with no 64-bit hash collisions;\n"
           "avg_probes/ins > 1 shows that compressing to size ~ n still collides --\n"
           "so emails need a COMPACT hash (probing), not a collision-free perfect hash.\n\n");

    // ---- Library baseline: radix-sort the 64-bit hashes (groups duplicates) ----
    // Sorting by hash gives a grouping/dedup order (NOT alphabetical), and is the
    // fast way to find duplicates in bulk. Timed for comparison.
    uint64_t *kin,*kout; void* tmp=nullptr; size_t tb=0;
    HIP_CHECK(hipMalloc(&kin, (size_t)n*sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&kout,(size_t)n*sizeof(uint64_t)));
    memcpy(kin, keys, (size_t)n*sizeof(uint64_t));
    HIP_CHECK((hipError_t)rocprim::radix_sort_keys(nullptr, tb, kin, kout, n));
    HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
    (void)rocprim::radix_sort_keys(tmp, tb, kin, kout, n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
    memcpy(kin, keys, (size_t)n*sizeof(uint64_t));
    hipEvent_t r0,r1; HIP_CHECK(hipEventCreate(&r0)); HIP_CHECK(hipEventCreate(&r1));
    HIP_CHECK(hipEventRecord(r0));
    (void)rocprim::radix_sort_keys(tmp, tb, kin, kout, n);
    HIP_CHECK(hipEventRecord(r1)); HIP_CHECK(hipEventSynchronize(r1));
    float lib_ms=0.f; HIP_CHECK(hipEventElapsedTime(&lib_ms, r0, r1));
    HIP_CHECK(hipEventDestroy(r0)); HIP_CHECK(hipEventDestroy(r1));
    bool sorted=true; int dups=0;
    for (int i=1;i<n;++i){ if(kout[i-1]>kout[i]) sorted=false; if(kout[i-1]==kout[i]) ++dups; }
    printf("Library radix sort of hashes: %.3f ms (%.1f Mkeys/s)  sorted=%s  adjacent_dups=%d\n",
           lib_ms, n/(lib_ms*1.0e3), sorted?"YES":"NO", dups);

    (void)hipFree(tmp);(void)hipFree(kin);(void)hipFree(kout);
    (void)hipFree(nbuf);(void)hipFree(keys);(void)hipFree(tkey);(void)hipFree(tcnt);
    (void)hipFree(probe);(void)hipFree(notfound);
    return 0;
}
