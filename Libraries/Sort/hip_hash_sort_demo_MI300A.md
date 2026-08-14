---
title: "Hands-On: Data-Aware Hash Sorting on MI300A (HIP)"
subtitle: "MPO Training Workshop — Sept 8, 2026"
author: "MPO Training"
date: "September 8, 2026"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

# Data-Aware Hash Sorting on MI300A — Hands-On Exercise

This exercise builds the two sorting examples from the libraries session as
runnable HIP code and scales them from a single MI300A APU up to a multi-node
run. You will:

1. Sort a **nationwide mailer by ZIP code** with a **counting sort** (non-unique
   keys), then contrast with a **true perfect hash** on **unique dense serials**.
2. Scale to **ZIP+4 / last names** where the key range is large or sparse, using a
   **compact hash**.
3. Run a **merge-free multi-node sort** (MPI + HIP) with distribution-aware
   partitioning.
4. **Exploit the MI300A APU's shared/unified memory** to eliminate host↔device
   copies (extra credit).
5. Compare the custom sorts against **library baselines** (thrust, rocPRIM, hipCUB)
   and learn the supported sort calls, including index sort / argsort (§3.7).
6. Profile everything (Appendix A).

> Build/PDF note: render with `pandoc hip_hash_sort_demo_MI300A.md -o demo.pdf`.

---

## 0. The MI300A APU: why "shared memory" changes the code

MI300A is an **APU**: the CPU (EPYC "Zen 4") cores and the CDNA3 GPU XCDs sit on
the same package and share **one physically unified pool of HBM3** with hardware
cache coherence. There is **no PCIe hop** and, importantly, **no separate device
copy of your data**.

Practical consequences for this exercise:

- A pointer from `hipMalloc` (or even ordinary `malloc`/`new` with XNACK on)
  is valid on **both** the CPU and the GPU. You generate data on the CPU and sort
  it on the GPU with **zero `hipMemcpy`**.

The **extra-credit** portions below show the copy-free pattern and mark where a
discrete-GPU code would have needed explicit transfers.

### Get an slurm allocation with a GPU

```bash
salloc -N 1 --ntasks=4 --cpus-per-task=1 --gpus=4 -t 01:00:00
```

# add `-p <slurm queue>` to request a particular 

### Environment setup

- Enable page migration / unified addressing:
- Load ROCm and OpenMPI environments with module load
- The makefile queries the arch (MI300A is `gfx942`) and compiles for it:
- Checks for GPU and gpu type with rocminfo and hipcc toolchain

```bash
export HSA_XNACK=1           # allow GPU to fault on host pages (unified memory)
module load rocm openmpi
rocminfo | grep -m1 gfx
hipcc --version              # confirm ROCm/clang toolchain
```

### Build everything (Makefile provided)

All sources, the name generator, and the build rules ship with this exercise. To
build the binaries and generate the sample data in one step:

```bash
make                 # builds zip_sort, name_sort, multinode_zip_sort; makes names.txt
make NCOUNT=20000    # regenerate a larger name list
make pdf             # render this document to PDF (needs pandoc + LaTeX)
make clean
```

The multi-node target needs a ROCm-aware MPI. On AAC, `module load rocm/7.2.4
openmpi` provides OpenMPI 5.0.10 and sets include and library paths:

```bash
module load rocm/7.2.4 openmpi
make multinode_zip_sort
```

To produce the PDF directly without make:

```bash
pandoc hip_hash_sort_demo_MI300A.md -o hip_hash_sort_demo_MI300A.pdf \
  --toc --highlight-style=tango -V colorlinks=true -V geometry:margin=1in
```

---

## 1. Example 1 — nationwide mailer, sort by ZIP (single APU)

A 5-digit ZIP is a bounded, known range `[0, 100000)`. Because many addresses
share a ZIP, keys are **not unique**, so the "one index per bucket" perfect hash
generalizes to a **counting (bucket) sort**: histogram → exclusive-scan offsets →
scatter. This is a single O(n) pass plus one scan — the pattern from the paper.

### 1.1 Reference code — `zip_sort.hip`

The custom counting sort (histogram → `thrust::exclusive_scan` → scatter) runs on
unified memory, and the program then sorts a **copy** of the same keys with the
`rocprim::radix_sort_keys` **library baseline** so you get a fair custom-vs-library
comparison in one run (both timed *warm*; measured numbers in §3.7.5). This is the
complete shipped file:

```cpp
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
    // rocPRIM uses the two-call temp-storage pattern: query size, then run.
    int *zip_in, *zip_out; void* tmp = nullptr; size_t tb = 0;
    HIP_CHECK(hipMalloc(&zip_in,  (size_t)n*sizeof(int)));
    HIP_CHECK(hipMalloc(&zip_out, (size_t)n*sizeof(int)));
    memcpy(zip_in, zip, (size_t)n*sizeof(int));
    HIP_CHECK((hipError_t)rocprim::radix_sort_keys(nullptr, tb, zip_in, zip_out, n)); // 1) query temp
    HIP_CHECK(hipMalloc(&tmp, tb ? tb : 1));
    (void)rocprim::radix_sort_keys(tmp, tb, zip_in, zip_out, n); HIP_CHECK(hipDeviceSynchronize()); // warm-up
    memcpy(zip_in, zip, (size_t)n*sizeof(int));

    hipEvent_t ls, le; HIP_CHECK(hipEventCreate(&ls)); HIP_CHECK(hipEventCreate(&le));
    HIP_CHECK(hipEventRecord(ls));
    (void)rocprim::radix_sort_keys(tmp, tb, zip_in, zip_out, n);                       // 2) run
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
```

### 1.2 Build & run (start small, then scale)

```bash
hipcc -O3 --offload-arch=gfx942 zip_sort.hip -o zip_sort
./zip_sort 1024          # tiny, for correctness
./zip_sort 1000000       # 1M records
./zip_sort 100000000     # 100M — fits easily in MI300A unified HBM
```

### 1.3 Exercises

- **E1.1** Confirm zero `hipMemcpy` calls (Appendix A shows how to verify with a
  trace). Compare against a version that uses separate `hipMalloc` +
  `hipMemcpy` and measure the difference on the APU.
- **E1.2 (LDS extra credit)** ZIP has 100k buckets (too big for LDS), but the
  *scatter* is atomic-heavy. Try a **per-block LDS-privatized histogram** for the
  low 3 digits and combine — where does contention actually hurt?
- **E1.3** Make the sort carry a payload (weight/route) and produce a per-ZIP count
  report for the mail run.
- **E1.4 (baseline)** `zip_sort` now also prints a `rocprim::radix_sort_keys` time on
  the same keys. Compare it to the counting-sort time and explain the gap (see
  §3.7 and E-lib.2).

---

## 1b. Example 1b — a *true* perfect hash: unique, dense keys

Example 1 is honestly a **counting sort**, not a perfect hash: many addresses share
a ZIP, so keys are not unique and we need histogram → scan → **atomic** scatter. A
genuine **perfect hash** (the paper's "one index per bucket") needs two guarantees:

1. a **dense, bounded** key range, `key ∈ [BASE, BASE+n)` with range `≈ n`, and
2. **unique** keys — each occurs **exactly once** (a bijection).

When both hold, placement is a **single write per element with no atomics, no
histogram, and no scan**:

```cpp
slot[key[i] - BASE] = i;   // every destination slot has exactly one writer
```

**Why not emails?** Emails are unique per recipient (guarantee 2) but live in an
enormous, sparse string space (they fail guarantee 1). Hashing an email to an
integer and compressing to a table of size `≈ n` reintroduces collisions, so emails
belong to the **compact-hash** family of Example 2 (`distinct == n`), not to a dense
perfect-hash scatter. To get guarantee 1 you need a key that is already a dense
integer — e.g. a **unique sequential serial number** assigned to each item.

### 1b.1 Reference solution — `id_sort.hip`

Real-world framing: a mailing batch where each mailpiece carries a **unique
sequential serial** (think USPS Intelligent Mail serial). The pieces come back from
the sorter shuffled; we reassemble canonical order by serial. `id_sort.hip` builds a
shuffled permutation of `[BASE, BASE+n)`, reorders it with the perfect-hash scatter,
compares against a general library radix sort, and verifies the result is an exact
permutation. Pass `dup=1` to inject a duplicate key and watch a record get lost.

```bash
hipcc -O3 --offload-arch=gfx942 id_sort.hip -o id_sort
./id_sort 1000000        # unique dense keys -> exact permutation
./id_sort 100000000      # 100M pieces
./id_sort 1000000 1      # inject a duplicate: uniqueness violated
```

### 1b.2 What it shows (validated on MI300A, ROCm 7.2.4)

Because the perfect hash skips the histogram, the exclusive scan, **and** the
atomics, it beats even a tuned library radix sort when the structure is known:

```text
# ./id_sort 100000000
  perfect-hash scatter : 2.891 ms (34588.7 Mkeys/s)  no atomics/scan
  library radix sort   : 7.652 ms (13068.0 Mkeys/s)  temp=816676868B
  library/perfect time ratio = 2.65  (>1 => perfect hash faster)
  result: exact permutation  (all 100000000 slots written, order verified)

# ./id_sort 1000000 1   (duplicate injected)
  result: NOT a bijection  unwritten_slots=1  order_ok=YES
  -> a duplicate key overwrote a slot and left another empty;
     this is exactly why the perfect hash requires UNIQUE keys.
```

The `dup=1` case is the whole lesson: with duplicate keys two writers target the
same slot, one write wins, and another slot is never written — a **silently lost
record**. That is precisely why ZIP (non-unique) must fall back to the counting sort
while unique serials can use the faster bijective scatter.

### 1b.3 Exercises

- **E1b.1** Compare `id_sort` and `zip_sort` at 1M / 100M. The perfect hash avoids
  the scan and the atomic scatter — quantify how much each of those costs by
  disabling them in `zip_sort`.
- **E1b.2** Run `./id_sort 1000000 1` and confirm exactly one slot is left unwritten.
  Add a device-side duplicate detector (e.g. `atomicCAS` on a per-slot "written"
  flag) that reports which serials collided.
- **E1b.3** Carry a payload: change `slot[d] = i` to also gather a record field, so
  the output is the fully reordered mailing (not just the permutation).
- **E1b.4 (discussion)** Emails are unique but sparse. Sketch how you would build a
  **minimal perfect hash function** offline to map a fixed email set bijectively to
  `[0,n)`, then reuse this exact scatter. What breaks if the set changes?

---

## 2. Example 2 — conference last names (sparse keys → compact hash)

Encode the first `k` letters of a last name as a base-26 integer key. For `k=1`
there are 26 buckets (heavily skewed: S/M/B/C common; Q/X/Z rare). For `k=2,3` the
key space is 676 / 17,576 but a few hundred attendees fill almost none of it — a
perfect-hash table would be mostly empty. Use a **compact hash**: table sized to
the attendee count, collisions resolved by **open addressing with quadratic
probing**.

### 2.1 Reference solution — `name_sort.hip`

The complete program runs three parts on the same unified-memory data:
**(A)** an LDS-privatized first-letter histogram for registration-desk balancing,
**(B)** the actual alphabetical sort via `thrust::sort_by_key` (rocPRIM radix under
the hood — the library workhorse), plus a **library baseline** that calls
`rocprim::radix_sort_pairs` and `hipcub::DeviceRadixSort::SortPairs` directly on the
same keys for a head-to-head timing, and **(C)** a compact hash (open addressing +
quadratic probing) built and verified over the large, sparse 64-bit name-key space.
Names are packed base-27 into a 64-bit key so numeric order equals lexicographic
order.

```cpp
// name_sort.hip  —  Example 2 reference solution (MI300A / gfx942)
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

// Pack up to KLETTERS letters base-27 (0 = pad => shorter prefixes sort first).
__host__ __device__ inline uint64_t pack_key(const char* s){
    uint64_t k = 0;
    for (int j = 0; j < KLETTERS; ++j){
        char ch = s[j]; int v = 0;
        if      (ch >= 'a' && ch <= 'z') v = ch - 'a' + 1;
        else if (ch >= 'A' && ch <= 'Z') v = ch - 'A' + 1;
        k = k * 27 + (uint64_t)v;
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
    if (i < n){ const char* s = names + (size_t)i*MAXLEN;
        keys[i] = pack_key(s); letter[i] = first_letter(s); }
}

// A) LDS-privatized histogram: 26 bins in LDS, one global atomic/bin/block.
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

// C) Compact hash: 2-universal compression + quadratic probing.
static const uint64_t EMPTY = 0xFFFFFFFFFFFFFFFFULL;
static const uint64_t CH_M = 2147483647ULL;   // 2^31-1 prime => no 64-bit overflow
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
    const char* path = (argc > 1) ? argv[1] : "names.txt";
    std::ifstream in(path);
    if (!in){ fprintf(stderr, "cannot open %s\n", path); return 1; }
    std::vector<std::string> names;
    for (std::string line; std::getline(in, line); ) if(!line.empty()) names.push_back(line);
    int n = (int)names.size();
    if (n == 0){ fprintf(stderr, "no names in %s\n", path); return 1; }

    // unified memory: CPU fills, GPU reads, no hipMemcpy
    char *nbuf; uint64_t *keys,*keys2; int *letter,*id,*gcount;
    HIP_CHECK(hipMalloc(&nbuf,(size_t)n*MAXLEN));
    HIP_CHECK(hipMalloc(&keys,(size_t)n*sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&keys2,(size_t)n*sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&letter,(size_t)n*sizeof(int)));
    HIP_CHECK(hipMalloc(&id,(size_t)n*sizeof(int)));
    HIP_CHECK(hipMalloc(&gcount,26*sizeof(int)));
    memset(nbuf,0,(size_t)n*MAXLEN);
    for (int i=0;i<n;++i){ strncpy(nbuf+(size_t)i*MAXLEN,names[i].c_str(),MAXLEN-1); id[i]=i; }
    for (int b=0;b<26;++b) gcount[b]=0;

    int T=256, B=(n+T-1)/T;
    compute_keys<<<B,T>>>(nbuf,n,keys,letter); HIP_CHECK(hipDeviceSynchronize());
    memcpy(keys2,keys,(size_t)n*sizeof(uint64_t));         // host copy for sorting

    // Part A — desk balancing
    hist_lds<<<B,T>>>(letter,n,gcount); HIP_CHECK(hipDeviceSynchronize());
    printf("First-letter distribution (n=%d):\n", n);
    for (int b=0;b<26;++b) if(gcount[b])
        printf("  %c: %5d  (%.1f%%)\n",'A'+b,gcount[b],100.0*gcount[b]/n);
    const int NDESK=4;
    printf("Suggested %d desks (balanced by frequency):\n",NDESK);
    { long target=(n+NDESK-1)/NDESK, run=0; int start=0,desk=0;
      for(int b=0;b<26;++b){ run+=gcount[b];
        if(run>=target||b==25){ printf("  Desk %d: %c-%c  (%ld packets)\n",
            desk+1,'A'+start,'A'+b,run); run=0; start=b+1; if(++desk==NDESK)break; } } }

    // Part B — alphabetical sort (rocPRIM radix via thrust)
    // Time only the sort itself (setup/allocation above is excluded).
    auto t0 = std::chrono::steady_clock::now();
    thrust::sort_by_key(thrust::device,keys2,keys2+n,id); HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    bool sorted=true;
    for(int i=1;i<n;++i) if(keys2[i-1]>keys2[i]){sorted=false;break;}
    printf("Alphabetical sort: %s  first=%s  last=%s\n", sorted?"OK":"FAILED",
           nbuf+(size_t)id[0]*MAXLEN, nbuf+(size_t)id[n-1]*MAXLEN);
    printf("  sort time: %.3f ms  (%.2f Mkeys/s)\n", sort_ms, n / (sort_ms * 1.0e3));

    // Part B baseline — the SAME 64-bit keys via all three libraries, timed the
    // same way (fresh copy + one warm-up). thrust::sort_by_key dispatches to
    // rocPRIM but allocates its scratch inside every call; rocPRIM and hipCUB use
    // the two-call temp-storage pattern (query size, allocate once, reuse).
    {
        uint64_t *kin,*kout; int *iin,*iout; void* tmp=nullptr; size_t tb=0;
        HIP_CHECK(hipMalloc(&kin,(size_t)n*sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&kout,(size_t)n*sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&iin,(size_t)n*sizeof(int)));
        HIP_CHECK(hipMalloc(&iout,(size_t)n*sizeof(int)));
        auto reset=[&]{ memcpy(kin,keys,(size_t)n*sizeof(uint64_t)); for(int i=0;i<n;++i) iin[i]=i; };
        auto el=[&](hipEvent_t a,hipEvent_t b){ float m=0.f; HIP_CHECK(hipEventElapsedTime(&m,a,b)); return m; };

        // thrust::sort_by_key (in-place on kin/iin), warmed up
        reset(); thrust::sort_by_key(thrust::device,kin,kin+n,iin); HIP_CHECK(hipDeviceSynchronize());
        reset(); hipEvent_t ta,t2; HIP_CHECK(hipEventCreate(&ta)); HIP_CHECK(hipEventCreate(&t2));
        HIP_CHECK(hipEventRecord(ta)); thrust::sort_by_key(thrust::device,kin,kin+n,iin);
        HIP_CHECK(hipEventRecord(t2)); HIP_CHECK(hipEventSynchronize(t2));
        float thr_ms=el(ta,t2); HIP_CHECK(hipEventDestroy(ta)); HIP_CHECK(hipEventDestroy(t2));

        // rocprim::radix_sort_pairs — two-call temp storage, warmed up
        reset(); HIP_CHECK((hipError_t)rocprim::radix_sort_pairs(nullptr,tb,kin,kout,iin,iout,n));
        HIP_CHECK(hipMalloc(&tmp,tb?tb:1));
        (void)rocprim::radix_sort_pairs(tmp,tb,kin,kout,iin,iout,n); HIP_CHECK(hipDeviceSynchronize());
        reset(); hipEvent_t ra,rb; HIP_CHECK(hipEventCreate(&ra)); HIP_CHECK(hipEventCreate(&rb));
        HIP_CHECK(hipEventRecord(ra)); (void)rocprim::radix_sort_pairs(tmp,tb,kin,kout,iin,iout,n);
        HIP_CHECK(hipEventRecord(rb)); HIP_CHECK(hipEventSynchronize(rb));
        float rocp_ms=el(ra,rb); HIP_CHECK(hipEventDestroy(ra)); HIP_CHECK(hipEventDestroy(rb)); HIP_CHECK(hipFree(tmp));

        // hipcub::DeviceRadixSort::SortPairs — two-call temp storage, warmed up
        tmp=nullptr; tb=0; reset();
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(nullptr,tb,kin,kout,iin,iout,n));
        HIP_CHECK(hipMalloc(&tmp,tb?tb:1));
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(tmp,tb,kin,kout,iin,iout,n)); HIP_CHECK(hipDeviceSynchronize());
        reset(); hipEvent_t ha,hb; HIP_CHECK(hipEventCreate(&ha)); HIP_CHECK(hipEventCreate(&hb));
        HIP_CHECK(hipEventRecord(ha)); HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(tmp,tb,kin,kout,iin,iout,n));
        HIP_CHECK(hipEventRecord(hb)); HIP_CHECK(hipEventSynchronize(hb));
        float cub_ms=el(ha,hb); HIP_CHECK(hipEventDestroy(ha)); HIP_CHECK(hipEventDestroy(hb)); HIP_CHECK(hipFree(tmp));

        printf("Library baseline (same keys, key-value radix sort, warmed):\n");
        printf("  thrust::sort_by_key      : %.3f ms  (%.2f Mkeys/s)  (scratch alloc'd internally)\n", thr_ms, n/(thr_ms*1.0e3));
        printf("  rocprim::radix_sort_pairs: %.3f ms  (%.2f Mkeys/s)\n", rocp_ms, n/(rocp_ms*1.0e3));
        printf("  hipcub::DeviceRadixSort  : %.3f ms  (%.2f Mkeys/s)\n", cub_ms, n/(cub_ms*1.0e3));
        (void)hipFree(kin);(void)hipFree(kout);(void)hipFree(iin);(void)hipFree(iout);
    }

    // Part C — compact hash build + verify
    int tableSize=(int)next_prime((long)(2*n+1));
    uint64_t* tkey; int* tcnt; unsigned long long* probe; int* notfound;
    HIP_CHECK(hipMalloc(&tkey,(size_t)tableSize*sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&tcnt,(size_t)tableSize*sizeof(int)));
    HIP_CHECK(hipMalloc(&probe,sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&notfound,sizeof(int)));
    for(int i=0;i<tableSize;++i){ tkey[i]=EMPTY; tcnt[i]=0; } *probe=0; *notfound=0;
    ch_insert<<<B,T>>>(keys,n,tkey,tcnt,tableSize,probe); HIP_CHECK(hipDeviceSynchronize());
    ch_query<<<B,T>>>(keys,n,tkey,tableSize,notfound);    HIP_CHECK(hipDeviceSynchronize());
    int occupied=0; for(int i=0;i<tableSize;++i) if(tkey[i]!=EMPTY) ++occupied;
    printf("Compact hash: tableSize=%d(prime) entries=%d distinct=%d\n",tableSize,n,occupied);
    printf("  load factor=%.3f  avg probes/insert=%.3f  query-miss=%d (expect 0)\n",
           (double)n/tableSize,(double)(*probe)/n,*notfound);

    (void)hipFree(nbuf);(void)hipFree(keys);(void)hipFree(keys2);(void)hipFree(letter);(void)hipFree(id);
    (void)hipFree(gcount);(void)hipFree(tkey);(void)hipFree(tcnt);(void)hipFree(probe);(void)hipFree(notfound);
    return 0;
}
```

> The three-library baseline between Part B and Part C prints
> `thrust::sort_by_key`, `rocprim::radix_sort_pairs`, and `hipcub::DeviceRadixSort`
> timings on the same 64-bit keys (measured numbers in §3.7.5, exercise E2.5).

### 2.2 Build & run

```bash
hipcc -O3 --offload-arch=gfx942 name_sort.hip -o name_sort
./name_sort
```

### 2.3 Exercises (extend the reference)

- **E2.1** The provided `names.txt` (from `gen_names.py`) matches real U.S.
  surname-initial frequencies. Run `./name_sort names.txt` and read off the
  frequency-weighted desk boundaries — do the four desks balance?
- **E2.2** Sweep the load factor: change `tableSize` to `1.1*n`, `1.5*n`, `2*n`,
  `4*n` and plot average probes/insert. Where does quadratic probing start to
  struggle?
- **E2.3** Replace quadratic probing with linear probing and observe primary
  clustering in the probe counts on the skewed name data.
- **E2.4 (discussion)** For `k=1` (26 keys) a dense counting sort wins; for the
  full 64-bit key the space is enormous and sparse so the compact hash is the
  memory-efficient structure. Explain the crossover in terms of load factor.
- **E2.5 (baseline)** Part B now prints `thrust::sort_by_key`,
  `rocprim::radix_sort_pairs`, and `hipcub::DeviceRadixSort` timings on the same
  64-bit keys. With the default 5000-name list these are microsecond-scale and
  launch-overhead-bound, so regenerate a large list first
  (`make names.txt NCOUNT=2000000`) to compare steady-state throughput (see §3.7).

---

## 2b. Example 2b — emails: unique but *sparse* keys (compact hash)

This is the case people usually reach for when they hear "unique keys" — and it is
exactly why a perfect hash does **not** apply. An email is **unique** per recipient
(guarantee 2 from §1b) but lives in a huge, sparse **string** space, so it fails the
**dense-range** guarantee 1. There is no `[0,n)` array to scatter into. Instead:

1. reduce each email string to a 64-bit key with a cheap device hash (FNV-1a), then
2. store the keys in a **compact hash** table sized `≈ n` (not `≈` the key range),
   resolving the collisions that the size-`n` compression creates by open addressing.

The crucial lesson: **even with 100% unique emails you still get collisions after
compression**, so you still probe. Uniqueness buys you `distinct == n` (no lost
records), *not* a collision-free perfect hash.

### 2b.1 Reference solution — `email_sort.hip`

`email_sort.hip` hashes each email to 64-bit (FNV-1a), builds the compact hash, and
**sweeps the load factor** to show probes vs. load. It also radix-sorts the hashes
as a bulk-dedup baseline (sorting by hash groups duplicates, but is *not*
alphabetical).

```bash
make emails.txt                 # gen_emails.py, 200k unique emails (ECOUNT to change)
hipcc -O3 --offload-arch=gfx942 email_sort.hip -o email_sort
./email_sort emails.txt
```

### 2b.2 What it shows (validated on MI300A, ROCm 7.2.4)

```text
Compact hash (open addressing + quadratic probing), unique keys:
  factor     tableSize    load      avg_probes/ins   distinct  miss
  1.10       220009       0.909     3.007            200000    0
  1.30       260003       0.769     2.080            200000    0
  1.50       300007       0.667     1.752            200000    0
  2.00       400009       0.500     1.437            200000    0
  3.00       600011       0.333     1.237            200000    0
  4.00       800011       0.250     1.161            200000    0
```

- `distinct == n` (200000) confirms the keys are unique and that **FNV-1a produced
  no 64-bit collisions** for this set (had two emails hashed equal, `distinct` would
  drop below `n`).
- `avg_probes/ins` climbs from ~1.16 at load 0.25 to ~3.0 at load 0.91 — the
  size-`n` compression collides, so you probe. This is the compact-hash cost curve
  that a dense perfect hash (id_sort, `key-base`) simply does not have.

### 2b.3 Where "a highly optimized hash" fits (and where it does not)

- The **dense perfect hash** (§1b) is `key - base` — a subtraction. It is
  **memory-bound on the scatter**, so a faster hash function changes nothing there.
- Here, in the **compact hash**, the *hash function* matters: a better-distributed
  hash lowers collisions/probes, and a cheaper one lowers per-key compute. FNV-1a is
  a fine cheap default; Murmur/xxHash-style finalizers are common alternatives.
- Google's fast hashing is mostly the wrong shape for this device path: Abseil
  **SwissTable** is a **CPU-SIMD hash *table***, not a GPU device hash, and
  FarmHash/HighwayHash are CPU-SIMD hash *functions*. They shine as a **CPU-side**
  dedup on the APU's Zen4 cores (shared memory, no copy), not as a substitute for
  the GPU compact hash. The GPU-native table comparison would be cuCollections
  (CUDA), which has no first-class ROCm drop-in — an instructive portability gap.

### 2b.4 Exercises

- **E2b.1** Read off the probes-vs-load curve above. Which load factor is the best
  memory/probe trade-off for your query mix?
- **E2b.2** Swap FNV-1a for a Murmur/xxHash-style 64-bit finalizer and compare
  `avg_probes/ins` and `distinct` (watch for any 64-bit collisions).
- **E2b.3** Replace quadratic probing with linear probing and observe primary
  clustering at high load factors on the email keys.
- **E2b.4 (APU)** Dedup the emails **on the CPU** with `std::unordered_map` (or
  `absl::flat_hash_map` if available) over the same unified buffer, and compare
  wall-clock and energy against the GPU compact hash. When is each the right tool?

---

## 3. Scaling out — merge-free multi-node sort (MPI + HIP)

Range-partition the key space across ranks so the global result is a simple
**concatenation** (no distributed merge): each rank owns a disjoint ZIP range,
we shuffle each record to its owner with one all-to-all, then sort locally.

### 3.1 Distribution-aware partitioning (assume → place → adjust)

Rather than a separate sampling pass:

1. Start with **assumed** range boundaries (uniform split of `[0,100000)` across
   ranks).
2. Route the **first ~10%** of local records for real.
3. **Adjust** boundaries from the observed histogram of that 10%.
4. **Move** only the small misplaced fraction; stream the remaining 90%.

### 3.2 Reference solution — `multinode_zip_sort.hip`

Complete MPI + HIP program. It samples the first 10% to adjust boundaries, shuffles
with `MPI_Alltoallv`, sorts locally on the GPU, and validates the merge-free result
(local order + seams + load balance). hipMalloc pointers are host-accessible on the
APU, so the MPI calls need no separate staging buffers. Pass `skew=1` as the second
argument to force an imbalanced input and observe the failure/repair.

```cpp
// multinode_zip_sort.hip  —  Example 3 reference solution (MPI + HIP, MI300A)
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
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) atomicAdd(&c[z[i]], 1); }
__global__ void scatter_vals(const int* z, int n, int* off, int* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){ int p = atomicAdd(&off[z[i]], 1); out[p] = z[i]; } }

static inline int owner_of(int zip, const std::vector<int>& bnd, int nranks){
    for (int r = 0; r < nranks; ++r) if (zip >= bnd[r] && zip < bnd[r+1]) return r;
    return nranks - 1;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Bind one APU per rank using the NODE-LOCAL rank (robust to any launch
    // mapping, and removes the need for an external ROCR_VISIBLE_DEVICES wrapper).
    MPI_Comm nodecomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &nodecomm);
    int local_rank; MPI_Comm_rank(nodecomm, &local_rank); MPI_Comm_free(&nodecomm);
    int ndev = 1; (void)hipGetDeviceCount(&ndev);
    int mydev = local_rank % ndev; HIP_CHECK(hipSetDevice(mydev));
    char pcibus[64]={0}; (void)hipDeviceGetPCIBusId(pcibus,(int)sizeof(pcibus),mydev);
    printf("[rank %d] local_rank=%d device=%d pci=%s (ndev=%d)\n",
           rank, local_rank, mydev, pcibus, ndev);

    long n_local = (argc > 1) ? atol(argv[1]) : (1L<<22);
    int  skew    = (argc > 2) ? atoi(argv[2]) : 0; // 1 => skewed input

    int* zip; HIP_CHECK(hipMalloc(&zip, (size_t)n_local*sizeof(int)));
    srand(1000 + rank);
    for (long i = 0; i < n_local; ++i)
        zip[i] = skew ? ((rand()%100 < 80) ? (rand()%10000) : (rand()%NB)) : (rand()%NB);

    // per-rank timers; barrier aligns the distributed-sort start
    MPI_Barrier(MPI_COMM_WORLD); double t_dist0 = MPI_Wtime();

    // 1) assume-place-adjust: build boundaries from a 10% global sample
    long sample = n_local/10; if (sample < 1) sample = n_local;
    std::vector<long> sh(NB,0);
    for (long i = 0; i < sample; ++i) sh[zip[i]]++;
    std::vector<long> gh(NB,0);
    MPI_Allreduce(sh.data(), gh.data(), NB, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    long total_sample = 0; for (int b=0;b<NB;++b) total_sample += gh[b];
    std::vector<int> bnd(nranks+1); bnd[0]=0; bnd[nranks]=NB;
    { long per=(total_sample+nranks-1)/nranks, run=0; int r=1;
      for (int b=0;b<NB && r<nranks;++b){ run+=gh[b]; if(run>=per){ bnd[r++]=b+1; run=0; } }
      for (; r<nranks; ++r) bnd[r]=NB; }

    // 2) partition all local records by owner; MPI_Alltoallv (data shuffle)
    double t_shuf0 = MPI_Wtime();
    std::vector<int> sendcounts(nranks,0);
    for (long i=0;i<n_local;++i) sendcounts[owner_of(zip[i],bnd,nranks)]++;
    std::vector<int> sdispls(nranks,0);
    for (int r=1;r<nranks;++r) sdispls[r]=sdispls[r-1]+sendcounts[r-1];
    std::vector<int> sendbuf(n_local), cursor(sdispls);
    for (long i=0;i<n_local;++i){ int r=owner_of(zip[i],bnd,nranks); sendbuf[cursor[r]++]=zip[i]; }
    std::vector<int> recvcounts(nranks,0);
    MPI_Alltoall(sendcounts.data(),1,MPI_INT,recvcounts.data(),1,MPI_INT,MPI_COMM_WORLD);
    std::vector<int> rdispls(nranks,0);
    for (int r=1;r<nranks;++r) rdispls[r]=rdispls[r-1]+recvcounts[r-1];
    long n_recv=(long)rdispls[nranks-1]+recvcounts[nranks-1];
    int* recvbuf; HIP_CHECK(hipMalloc(&recvbuf,(size_t)(n_recv>0?n_recv:1)*sizeof(int)));
    MPI_Alltoallv(sendbuf.data(),sendcounts.data(),sdispls.data(),MPI_INT,
                  recvbuf,recvcounts.data(),rdispls.data(),MPI_INT,MPI_COMM_WORLD);
    double t_shuffle = MPI_Wtime() - t_shuf0;

    // 3) local GPU counting sort of received records
    double t_ls0 = MPI_Wtime();
    int *counts,*offsets,*out;
    HIP_CHECK(hipMalloc(&counts,NB*sizeof(int)));
    HIP_CHECK(hipMalloc(&offsets,NB*sizeof(int)));
    HIP_CHECK(hipMalloc(&out,(size_t)(n_recv>0?n_recv:1)*sizeof(int)));
    for (int b=0;b<NB;++b) counts[b]=0;
    if (n_recv>0){ int T=256,B=(int)((n_recv+T-1)/T);
        histogram<<<B,T>>>(recvbuf,(int)n_recv,counts); HIP_CHECK(hipDeviceSynchronize());
        thrust::exclusive_scan(thrust::device,counts,counts+NB,offsets); HIP_CHECK(hipDeviceSynchronize());
        scatter_vals<<<B,T>>>(recvbuf,(int)n_recv,offsets,out); HIP_CHECK(hipDeviceSynchronize()); }
    double t_local = MPI_Wtime() - t_ls0;
    bool local_ok=true; for(long i=1;i<n_recv;++i) if(out[i-1]>out[i]){local_ok=false;break;}

    // 4) merge-free: global offsets + seam check
    long gofs=0; MPI_Exscan(&n_recv,&gofs,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if (rank==0) gofs=0;
    int my_min=(n_recv>0)?out[0]:NB, my_max=(n_recv>0)?out[n_recv-1]:-1, prev_max=-1;
    if (rank>0)        MPI_Recv(&prev_max,1,MPI_INT,rank-1,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    if (rank<nranks-1) MPI_Send(&my_max, 1,MPI_INT,rank+1,7,MPI_COMM_WORLD);
    bool seam_ok=(rank==0)||(n_recv==0)||(prev_max<=my_min);
    double t_dist = MPI_Wtime() - t_dist0;

    long gmax=0,gsum=0; MPI_Reduce(&n_recv,&gmax,1,MPI_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_recv,&gsum,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    int lo=local_ok?1:0, so=seam_ok?1:0, glo,gso;
    MPI_Reduce(&lo,&glo,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&so,&gso,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    double gts,gtl,gtd;   // slowest rank per phase => phase wall-clock
    MPI_Reduce(&t_shuffle,&gts,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&t_local,  &gtl,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&t_dist,   &gtd,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    printf("[rank %d] owns [%d,%d) recv=%ld goff=%ld local=%s seam=%s"
           " | shuffle=%.3f ms local_sort=%.3f ms dist_total=%.3f ms\n",
           rank,bnd[rank],bnd[rank+1],n_recv,gofs,local_ok?"YES":"NO",seam_ok?"YES":"NO",
           t_shuffle*1e3,t_local*1e3,t_dist*1e3);
    if (rank==0){ double bal=(gsum>0)?(double)gmax/((double)gsum/nranks):1.0;
        printf("GLOBAL: total=%ld imbalance(max/avg)=%.2f local=%s seams=%s (skew=%d)\n",
               gsum,bal,glo?"ALL-OK":"FAIL",gso?"ALL-OK":"FAIL",skew);
        printf("GLOBAL: max times (ms): shuffle=%.3f local_sort=%.3f dist_total=%.3f\n",
               gts*1e3,gtl*1e3,gtd*1e3); }

    (void)hipFree(zip);(void)hipFree(recvbuf);(void)hipFree(counts);
    (void)hipFree(offsets);(void)hipFree(out);
    MPI_Finalize(); return 0;
}
```

### 3.3 Interactive SLURM run — single node, 4 APUs

For quick experiments (correctness checks, small sweeps) it is often easier to
grab an **interactive** allocation than to submit the batch script. Request one
MI300A node with all 4 APUs, then launch 4 ranks — one per APU — with `mpirun`
directly from the interactive shell (again `mpirun`, **not** `srun`).

```bash
# 1) Grab one node with its 4 APUs for an hour (add -p <queue> as needed)
salloc -N 1 --ntasks-per-node=4 --gpus-per-node=4 \
       -p PPAC_MI300A_SPX --exclusive -t 01:00:00

# 2) When the prompt returns on the allocated compute node:
module load rocm/7.2.4 openmpi
export HSA_XNACK=1

# 3) 4 ranks, one per APU.  Args: [records_per_rank] [skew(0|1)]
mpirun -np 4 --map-by ppr:4:node -x HSA_XNACK \
       ./multinode_zip_sort 4000000 0      # uniform keys
mpirun -np 4 --map-by ppr:4:node -x HSA_XNACK \
       ./multinode_zip_sort 4000000 1      # skewed keys (watch the repair)
```

The same node-local binding logic used by the batch path applies here: each rank
derives its `local_rank` from an `MPI_COMM_TYPE_SHARED` sub-communicator and calls
`hipSetDevice(local_rank % ndev)`, so the four ranks bind to APUs `0,1,2,3` and
print their device / PCI-bus id at startup — no `ROCR_VISIBLE_DEVICES` wrapper is
needed. Because you stay on the allocated node, you can iterate quickly (rebuild
with `make multinode_zip_sort`, rerun `mpirun`) without resubmitting. Type `exit`
to release the allocation when you are done.

### 3.4 SLURM launch on MI300A — `run_multinode.sbatch`

MI300A nodes expose 4 APUs. Launch one rank per APU with **`mpirun`, not `srun`**
(on this system the `srun --mpi=pmix` step launcher times out; `mpirun` is the
supported path). The batch script runs on the first allocated compute node, so
`mpirun` places the ranks from there.

**Do the ranks land on different GPUs, and is a wrapper needed?** No wrapper is
required. The program derives a **node-local rank** from an
`MPI_COMM_TYPE_SHARED` sub-communicator and calls `hipSetDevice(local_rank %
ndev)`, so on each node ranks bind to devices `0,1,2,3` (one APU each). It prints
its device and PCI-bus id at startup so you can confirm the mapping — e.g. on
one node the four APUs appear as `0000:01:00.0/.1/.2/.3`. This holds on 2 nodes ×
4 ranks too: each node runs its own shared-memory communicator, so the 4 ranks
per node get local ranks `0..3` regardless of the global rank numbering. (An
external `ROCR_VISIBLE_DEVICES` wrapper is an alternative, but is unnecessary
because the binding is done in code.)

```bash
#!/bin/bash
#SBATCH --job-name=zipsort
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4        # 4 APUs per node (SPX mode)
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#
# Usage: sbatch run_multinode.sbatch [records_per_rank] [skew(0|1)]

module load rocm/7.2.4 openmpi
export HSA_XNACK=1

NP=${SLURM_NTASKS:-8}
mpirun -np "$NP" --map-by ppr:4:node -x HSA_XNACK \
       ./multinode_zip_sort "${1:-4000000}" "${2:-0}"
```

```bash
# Single node (4 ranks): mpirun launches all ranks locally on the node
sbatch --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 run_multinode.sbatch 4000000 0
sbatch --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 run_multinode.sbatch 4000000 1

# Two nodes (8 ranks)
sbatch run_multinode.sbatch 4000000 0
```

**Validated output (4 ranks × 4M records, MI300A / ROCm 7.2.4):**

```text
# UNIFORM (skew=0)
[rank 0] owns ZIP [0,24983)     recv=3997879  global_offset=0         local_sorted=YES seam=YES
[rank 1] owns ZIP [24983,49996) recv=4000001  global_offset=3997879   local_sorted=YES seam=YES
[rank 2] owns ZIP [49996,75008) recv=4002866  global_offset=7997880   local_sorted=YES seam=YES
[rank 3] owns ZIP [75008,100000)recv=3999254  global_offset=12000746  local_sorted=YES seam=YES
GLOBAL: total=16000000  imbalance(max/avg)=1.00  local_sorted=ALL-OK  seams=ALL-OK  (skew=0)

# SKEWED (skew=1): 80% of keys in the low ZIP band -> boundaries shrink there
[rank 0] owns ZIP [0,3050)      recv=4004284  global_offset=0         local_sorted=YES seam=YES
[rank 1] owns ZIP [3050,6099)   recv=3997214  global_offset=4004284   local_sorted=YES seam=YES
[rank 2] owns ZIP [6099,9148)   recv=4002057  global_offset=8001498   local_sorted=YES seam=YES
[rank 3] owns ZIP [9148,100000) recv=3996445  global_offset=12003555  local_sorted=YES seam=YES
GLOBAL: total=16000000  imbalance(max/avg)=1.00  local_sorted=ALL-OK  seams=ALL-OK  (skew=1)
```

Note how under heavy skew the assume-place-adjust step shrinks the low-range
partitions (rank 0 owns only `[0,3050)`) so per-rank counts stay balanced
(imbalance ≈ 1.00), and the disjoint ordered ranges make the global merge a
no-op (verified seams + global offsets).

### 3.5 Exercises (extend the reference)

- **E3.1** Run with `skew=0` then `skew=1`; compare the reported
  `imbalance(max/avg)`. Then **disable** the boundary adjustment (force uniform
  `bnd[]`) and rerun `skew=1` to reproduce the **degenerate all-on-one-rank** case.
- **E3.2** Log the number of records that changed owner after the 10% calibration
  vs. the uniform assumption (add a counter around the `owner_of` reassignment).
- **E3.3** Swap the ZIP key for a **last-name** key and reuse the compact hash from
  Example 2 as the per-rank local sort.
- **E3.4 (scaling study)** Run 1 → 2 → 4 → 8 nodes; report records/sec and the
  fraction of wall time in `MPI_Alltoallv` (time it around the call).

---

## 3.6 (Optional, advanced) The radix memory story: rocPRIM Onesweep vs. Orochi's circular buffer

This ties the data-aware hash-sort memory trade-off back to the *general-purpose*
radix sort. Recall the throughline: a fast sort often pays for speed with memory
that scales with the wrong quantity, and the fix is to decouple the allocation.

**Part A — measure it on the stock ROCm stack (`rocprim_tempsize.hip`).**
rocPRIM's `device_radix_sort` uses Onesweep with decoupled look-back; its
temporary buffer (the look-back state) grows with the number of keys `n`. You can
see this *without sorting* by asking rocPRIM for its required temporary storage
(pass a null `temporary_storage` pointer):

```cpp
size_t temp_bytes = 0;
rocprim::radix_sort_keys(nullptr, temp_bytes, d_in, d_out, n);  // query only
printf("n=%zu  temp=%zu B  (%.3f B/key)\n", n, temp_bytes, (double)temp_bytes/n);
```

Build and run:

```bash
hipcc -O3 --offload-arch=gfx942 rocprim_tempsize.hip -o rocprim_tempsize
srun -p PPAC_MI300A_SPX -N1 -n1 --gpus=1 -t 4 ./rocprim_tempsize
```

Measured on MI300A (ROCm 7.2.4):

```
      n (keys)  temp_storage (B)   bytes/key
       1000000           4003912       4.004
      10000000          41260548       4.126
     100000000         412511236       4.125
     500000000        2062511108       4.125
    1000000000        4125010948       4.125
```

The scratch grows **linearly at ~4.125 bytes/key** — a billion keys needs ~4.1 GB
of temporary storage on top of the data itself.

**Part B — read the fix in Orochi (`ParallelPrimitives`), and port it to wave64.**
The GPUOpen circular-buffer extension (Kao & Yoshimura, *GPU Zen 3*, 2025) holds
that look-back buffer at a **constant ~2 MB regardless of `n`**. Clone Orochi and
read the mechanism — the `tail iterator`, `L_lookback`, and `N_table` in:

```bash
git clone --depth 1 https://github.com/GPUOpen-LibrariesAndSDKs/Orochi.git
$EDITOR Orochi/ParallelPrimitives/RadixSortKernels.h   # circular-buffer look-back
$EDITOR Orochi/ParallelPrimitives/RadixSort.cpp        # host driver + tail iterator
```

Orochi is *not* a ROCm library; it is host-side driver-API code that loads HIP at
runtime and compiles its kernels with hiprtc. It builds with plain `g++` (no
`hipcc`):

```bash
g++ -O3 -std=c++17 -I. "-D__debugbreak()=abort()" -include cstdlib \
    Orochi/Orochi.cpp Orochi/OrochiUtils.cpp \
    contrib/hipew/src/hipew.cpp contrib/cuew/src/cuew.cpp \
    ParallelPrimitives/RadixSort.cpp Test/RadixSort/main.cpp \
    -ldl -lpthread -o build/RadixSortTest
```

**The instructive part: out of the box it is wave32-only.** Orochi's radix sort
comes from the RDNA rendering/ray-tracing side, and `RadixSortConfigs.h` hardcodes
`WARP_SIZE = 32`. That is correct on the **RDNA Radeon workstation/consumer GPUs
(32-lane wavefront)** where it is designed to run. The **CDNA Instinct data-center
parts (MI300A = CDNA3) use a 64-lane wavefront**, so the multi-pass Onesweep reorder
indexes out of bounds (memory fault / wrong result for `n ≥ 3072`, the point at
which it leaves the single-pass kernel). It is a **wave32-vs-wave64 portability
gap, not a bug and not a ROCm-version issue.**

**Porting it to wave64 (`orochi_wave64.patch`, provided).** The change is small and
mechanical, and it is the whole lesson in miniature:

1. `WARP_SIZE = 64` (and derive `REORDER_NUMBER_OF_ITEM_PER_THREAD` from
   `WARP_SIZE`, not a hardcoded `/32`, so the per-lane arrays and loop counts match
   the wave).
2. In the reorder kernel's ballot-based warp match, widen the **lane mask to 64
   bits**: `__ballot` returns a 64-bit mask on wave64, so `broThreads` becomes
   `u64`, `(1u << lane)` becomes `((u64)1 << lane)`, and `__popc`/`__ffs` become
   `__popcll`/`__ffsll`.

That is it — no shuffle-scan iteration to add, because this implementation does its
intra-warp ranking with `__ballot` + `popcount` rather than a log-step shuffle scan.

Apply and rebuild:

```bash
cd Orochi && git apply orochi_wave64.patch
# rebuild RadixSortTest as above; run from a dir where ../ParallelPrimitives resolves
```

**Validated on MI300A (`gfx942`), ROCm 7.2.4** — the circular-buffer Onesweep path
now sorts correctly at every size and the bundled perf test passes:

```
n=1000  1e5  1e6  1e7   -> sorted=YES, mismatches=0 (all)
16.0 Mkeys, 32-bit:  ~6.6 GKeys/s (key-value),  ~7.0 GKeys/s (key-only)
16.0 Mkeys, 16-bit:  ~11.5 GKeys/s (key-value), ~12.1 GKeys/s (key-only)
```

> Caveat: this was validated under **ROCm 7.2.4** (the workshop stack), where hiprtc
> compiles the kernels quickly. Under ROCm 6.3.0 the *runtime* hiprtc compile of the
> patched kernels stalls (an unrelated JIT issue on that older driver); the offline
> `hipcc` compile is clean. Use 7.2.4 for this exercise.

**Exercises**
- **E3.5** Plot `temp_storage(n)` from Part A and fit the slope; confirm it is O(n).
  At what `n` does the scratch exceed your per-APU HBM budget alongside the data?
- **E3.6** In `RadixSortKernels.h`, identify where the circular buffer bounds the
  look-back distance (`L_lookback < N_table`) and explain, in two sentences, why
  that makes the temporal memory independent of `n`.
- **E3.7 (port it yourself)** Starting from stock Orochi, reproduce the wave64 port:
  find every place a 32-lane assumption hides (the `WARP_SIZE` constant, the `/32`
  divisor, and the 32-bit `__ballot`/`__popc`/`__ffs`/`(1u<<lane)` in the reorder
  kernel) and fix them. Confirm correctness with the bundled test. This is the
  canonical "RDNA-tuned kernel → CDNA wave64" migration in ~30 lines.

---

## 3.7 Library sort APIs — thrust, rocPRIM, hipCUB (baseline & reference)

The custom sorts above are *data-aware*: `zip_sort` exploits the bounded ZIP range
with a counting sort, and the multi-node code range-partitions so the merge is
free. To know whether that hand-written work is worth it, you need a **baseline**:
what does a stock library radix sort do on the same keys? ROCm ships three ways to
call one:

- **thrust** — the portable C++ STL-like API (`thrust::sort`, `thrust::sort_by_key`).
  On ROCm it dispatches to rocPRIM under the hood.
- **rocPRIM** — the AMD device-primitive library (`rocprim::radix_sort_keys`,
  `rocprim::radix_sort_pairs`). Lowest-level, most tuning knobs.
- **hipCUB** — a CUB-compatible wrapper over rocPRIM
  (`hipcub::DeviceRadixSort::SortKeys`/`SortPairs`). Use it when porting CUDA code
  that already calls CUB.

### 3.7.1 The two-call temporary-storage pattern (rocPRIM & hipCUB)

Unlike thrust, rocPRIM and hipCUB do not allocate scratch for you. You call the
function **twice**: first with a null storage pointer to learn the size, then with
a real allocation to do the work.

```cpp
size_t temp_bytes = 0; void* d_temp = nullptr;
// 1) query: temporary_storage == nullptr just fills temp_bytes
rocprim::radix_sort_keys(nullptr, temp_bytes, d_in, d_out, n);
hipMalloc(&d_temp, temp_bytes);
// 2) run: same arguments, real storage
rocprim::radix_sort_keys(d_temp, temp_bytes, d_in, d_out, n);
```

hipCUB is identical (`hipcub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, ...)`
then again with `d_temp`). thrust needs neither call — it manages scratch
internally (which is why it is the easiest but least controllable).

### 3.7.2 Side-by-side example — `sort_apis.hip`

`sort_apis.hip` sorts a **copy of the same array** with all three libraries
(keys-only, key-value pairs, and descending), verifies they agree bit-for-bit, and
prints hipEvent timings so you can compare them directly:

```bash
hipcc -O3 --offload-arch=gfx942 sort_apis.hip -o sort_apis
./sort_apis 4194304
```

Measured on MI300A (ROCm 7.2.4), `./sort_apis 1000000`:

```text
operation / library                  time(ms)      Mkeys/s
keys  thrust::sort                      0.753      1327.37
keys  rocprim::radix_sort_keys          0.260      3852.63   temp=4003912B
keys  hipcub::DeviceRadixSort           0.256      3899.49   temp=4003912B

pairs thrust::sort_by_key               0.977      1023.36
pairs rocprim::radix_sort_pairs         0.486      2057.93   temp=8003912B
pairs hipcub::DeviceRadixSort           0.497      2011.07   temp=8003912B

keys  thrust::sort (descending)         0.561      1782.13
keys  rocprim::radix_sort_keys_desc     0.293      3411.99   temp=4003912B
keys  hipcub::...SortKeysDescending     0.356      2808.01   temp=4003912B
```

All three agree bit-for-bit. rocPRIM and hipCUB (which wraps rocPRIM) are ~3x faster
than thrust here because thrust allocates its scratch inside every call while the
rocPRIM/hipCUB temp buffer is allocated once and reused — the same effect you see in
the name baseline below.

### 3.7.3 Supported sort calls (quick reference)

| Operation | thrust | rocPRIM | hipCUB |
|---|---|---|---|
| Keys, ascending | `thrust::sort` | `rocprim::radix_sort_keys` | `hipcub::DeviceRadixSort::SortKeys` |
| Keys, descending | `thrust::sort(..., thrust::greater<>())` | `rocprim::radix_sort_keys_desc` | `hipcub::DeviceRadixSort::SortKeysDescending` |
| Key+value pairs | `thrust::sort_by_key` | `rocprim::radix_sort_pairs` | `hipcub::DeviceRadixSort::SortPairs` |
| Pairs, descending | `thrust::sort_by_key(..., thrust::greater<>())` | `rocprim::radix_sort_pairs_desc` | `hipcub::DeviceRadixSort::SortPairsDescending` |
| Stable / comparison sort | `thrust::stable_sort` | `rocprim::merge_sort` | `hipcub::DeviceMergeSort::SortKeys` |
| Segmented (per-segment) | — (do it per range) | `rocprim::segmented_radix_sort_keys` | `hipcub::DeviceSegmentedRadixSort::SortKeys` |
| Bit-range subset | — | `begin_bit` / `end_bit` args | `begin_bit` / `end_bit` args |
| Temp storage | managed internally | two-call (`nullptr` then alloc) | two-call (`nullptr` then alloc) |

Notes:

- Radix sorts (`radix_sort_*` / `DeviceRadixSort`) are for unsigned/signed integer
  and floating-point keys; they are stable and usually fastest. Use the merge-sort
  variants for a custom comparator on arbitrary types.
- Restricting `begin_bit`/`end_bit` to only the significant bits (e.g. a ZIP fits
  in 17 bits, not 32) can cut radix passes — a cheap win (exercise E-lib.3).
- rocPRIM and hipCUB are **header-only** here: `#include <rocprim/rocprim.hpp>` /
  `#include <hipcub/hipcub.hpp>` and build with plain `hipcc`, no extra `-l` flag.

### 3.7.4 Index sort (argsort) vs. sorting the data — `index_sort.hip`

Frequently you do **not** want to move the records — you want a *permutation* that
orders them. That is an **index sort** (argsort): sort an index array `[0,1,2,...]`
by the keys, leave the payload in place, then either read records indirectly or
**gather** once to materialize a sorted copy. `name_sort.hip` already does this
implicitly (`thrust::sort_by_key(keys, keys+n, id)` sorts `id[]`, not the names).

`index_sort.hip` makes the trade-off explicit and measured:

```bash
hipcc -O3 --offload-arch=gfx942 index_sort.hip -o index_sort
./index_sort 4194304 8      # 4M records, 8-word (32 B) payload
```

- **Index sort**: sort `(key, 4-byte index)` pairs (rocPRIM / hipCUB / thrust),
  then a `gather` kernel copies each record once into sorted order.
- **Data sort**: sort `(key, payload)` directly, dragging the payload through every
  radix pass.

Measured on MI300A (ROCm 7.2.4), `./index_sort 1000000 8` (8-word / 32 B payload):

```text
strategy                                   time(ms)      Mkeys/s
1a idx rocprim pairs (sort only)             0.524        1908.38
   + gather payload = total                   0.677        1477.88
1b idx hipcub pairs (sort only)               0.482        2075.36
1c idx thrust::sort_by_key (sort only)        0.951        1051.51

2  data rocprim pairs (1 word payload)        0.321        3119.51
```

With a wide payload the index-sort + single gather wins because only 4 B/key move
through the passes; with a one-word payload, sorting the data directly is simplest.
Sweep the payload size to find the crossover (exercise E-lib.4).

### 3.7.5 Baselines already built into Examples 1 & 2

You do not need a separate program to see custom-vs-library — both reference
solutions now print a library baseline on the same data:

- `zip_sort` prints its **counting-sort** time and then a `rocprim::radix_sort_keys`
  time on a copy of the ZIP keys, plus the time ratio (both measured *warm*, after a
  histogram warm-up, so the comparison is fair). Measured on MI300A (ROCm 7.2.4):

```text
        n     custom counting    library radix   ratio (lib/custom)
  1000000     0.314 ms            0.266 ms        0.85   (library a bit faster)
  4000000     0.431 ms            0.431 ms        1.00   (tie)
100000000     7.494 ms            2.786 ms        0.37   (library ~2.7x faster)
```

  The result is instructive: on *uniformly random* ZIPs over 100k buckets the general
  radix sort is competitive at small `n` and actually **faster** at large `n`,
  because the counting sort's scatter is a storm of atomics into 100k HBM buckets. The
  custom sort pays off when the range is small (few buckets, atomics stay cheap) or
  when you also need the per-bucket histogram/offsets it produces for free — exactly
  the multi-node partitioning in §3.
- `name_sort` Part B prints all three library timings on the same 64-bit name keys,
  measured the same way (fresh copy + warm-up). Measured on MI300A (ROCm 7.2.4) with a
  2M-name list (`make names.txt NCOUNT=2000000`):

```text
  thrust::sort_by_key      : 1.831 ms  (1092.33 Mkeys/s)   (scratch alloc'd internally)
  rocprim::radix_sort_pairs: 0.675 ms  (2965.04 Mkeys/s)
  hipcub::DeviceRadixSort  : 0.619 ms  (3233.28 Mkeys/s)
```

  thrust is ~2.7x slower not because its sort is worse (it dispatches to rocPRIM) but
  because it allocates its temporary scratch inside every call; rocPRIM/hipCUB let you
  allocate the temp once and reuse it.

### 3.7.6 Exercises

- **E-lib.1** Run `./sort_apis` at 1M, 16M, 64M keys. Do the three libraries agree?
  Which is fastest, and does the ranking change with `n`?
- **E-lib.2** In `zip_sort`, compare the counting-sort time to the
  `rocprim::radix_sort_keys` baseline across `n` = 1M, 4M, 100M. At what `n` does the
  general radix sort overtake the counting sort, and why? (Hint: the scatter is
  atomic-bound into 100k HBM buckets.) When would the counting sort still be the
  right choice despite being slower here?
- **E-lib.3** In `sort_apis`, add `begin_bit=0, end_bit=17` to the rocPRIM/hipCUB
  ZIP-range call (keys < 2^17) and measure the speedup from fewer radix passes.
- **E-lib.4** In `index_sort`, sweep the payload from 1 to 64 words and plot
  index-sort+gather vs. data-sort time; identify the crossover.
- **E-lib.5** Swap `name_sort`'s `thrust::sort_by_key` for the direct
  `rocprim::radix_sort_pairs` baseline and confirm identical alphabetical output.

---

## 4. What "good" looks like

- Single APU, `zip_sort` at 100M records: dominated by the histogram/scatter
  atomics and the scan; **no** copy time (verify in the trace).
- Multi-node: after E3.1, per-rank counts within a few percent of each other; the
  merge step is a no-op (just offset bookkeeping).
- Extra credit achieved when your code has **zero `hipMemcpy`** and still validates.

---

# Appendix A — On-Your-Own Profiling

All commands assume `module load rocm` and the binaries built above. Do these in
order; each answers a specific question.

## A.1 Confirm the copy-free APU path (rocprofv3 trace)

Capture an application trace and confirm there are **no HtoD/DtoH copies**:

```bash
rocprofv3 --sys-trace -- ./zip_sort 100000000
# Inspect the generated trace (Perfetto/`.pftrace` or CSV). Look for:
#   - COPY (MEMCPY) rows  -> should be absent/negligible on the APU path
#   - kernel rows: histogram, scatter, and the thrust scan
```

**Task:** rebuild a `hipMemcpy` variant and compare the trace — quantify the copy
time you eliminated by using unified memory.

## A.2 Per-kernel counters (rocprofv3)

```bash
rocprofv3 --hip-trace --kernel-trace --stats -- ./zip_sort 100000000
```

**Task:** which kernel dominates — `histogram`, `scatter`, or the scan? Record its
duration and occupancy.

## A.3 Kernel analysis & roofline (rocprof-compute)

```bash
# Profile
rocprof-compute profile -n zipsort --  ./zip_sort 100000000
# Analyze (roofline + memory-chart)
rocprof-compute analyze -p workloads/zipsort/MI300A/
```

**Tasks:**
- Place the `scatter` kernel on the **roofline** — is it memory-bound (expected,
  atomics into HBM) or compute-bound?
- Note the L2/HBM traffic. Given MI300A's unified HBM, does the histogram's atomic
  pattern saturate bandwidth?

## A.4 System-level trace + power/energy (rocprof-sys)

`rocprof-sys` samples `amd-smi`/`rocm-smi` counters, so it doubles as your
**user-level power monitor** (no omnistat needed):

```bash
rocprof-sys-instrument -o zipsort.inst -- ./zip_sort 100000000
rocprof-sys-run -- ./zipsort.inst 100000000
# In the config, enable rocm-smi sampling for power (W) and energy (J):
#   ROCPROFSYS_USE_ROCM_SMI=ON
#   ROCPROFSYS_SAMPLING_GPUS=ON
```

**Tasks:**
- Plot GPU/APU **power (W)** over the run; integrate to **energy (J)** for the sort.
- Compare energy of the unified-memory path vs. the `hipMemcpy` path.

## A.5 Power of a SLURM-launched multi-node job

Sample power on each allocated node while the `mpirun` job runs. Add this to
`run_multinode.sbatch` just before the `mpirun` line:

```bash
# Start a background power sampler on every node in the allocation, then run.
# (Use a single-task-per-node srun ONLY to fan out the sampler, not the job.)
srun --ntasks-per-node=1 bash -c \
  'amd-smi monitor --power --interval 1 --loop > power_$(hostname).log 2>&1 &'

mpirun -np "$NP" --map-by ppr:4:node -x HSA_XNACK ./multinode_zip_sort 4000000 0
```

**Tasks:**
- Sum node-level average power × wall time for a **whole-job energy** estimate.
- Correlate power dips with the all-to-all shuffle phase (E3.4 timing).

## A.6 CPU-side profiling (uProf / Linux tools)

Because the APU shares memory, the CPU generation phase matters too:

```bash
# AMD uProf (CPU hotspots / memory):
AMDuProfCLI collect --config tbp -o uprof_out ./zip_sort 100000000
# Linux cache behavior of the CPU fill loop:
valgrind --tool=cachegrind ./zip_sort 1000000
perf stat -d ./zip_sort 10000000
```

**Task:** how much of end-to-end time is the CPU fill vs. GPU sort? Would
prefetch/`hipMemAdvise` hints change the balance on the APU?

## A.7 Profiling deliverable

Produce a one-page summary: dominant kernel, roofline position, energy for the
sort, and (multi-node) the shuffle fraction and per-rank balance after
assume-place-adjust.

---

## Appendix B — File manifest

| File | Purpose |
|---|---|
| `zip_sort.hip` | Example 1: counting sort by ZIP (non-unique keys) + library baseline |
| `id_sort.hip` | Example 1b: true perfect-hash scatter for unique dense keys (+ dup detection) |
| `name_sort.hip` | Example 2: LDS histogram + radix sort + compact hash (complete) |
| `email_sort.hip` | Example 2b: unique-but-sparse emails -> compact hash, load-factor/probe sweep |
| `multinode_zip_sort.hip` | Example 3: MPI range-partition + local sort (complete) |
| `rocprim_tempsize.hip` | §3.5: measures rocPRIM Onesweep temp-storage growth (validated) |
| `sort_apis.hip` | §3.7: same array sorted via thrust / rocPRIM / hipCUB (keys, pairs, descending) |
| `index_sort.hip` | §3.7.4: index sort (argsort) + gather vs. sorting the data directly |
| `orochi_wave64.patch` | §3.5 Part B: wave32→wave64 port of Orochi's radix sort (validated on MI300A, ROCm 7.2.4) |
| `run_multinode.sbatch` | SLURM launcher (mpirun), 1 rank/APU on MI300A |
| `gen_names.py` | Generates `names.txt` with realistic surname-initial frequencies |
| `names.txt` | Sample attendee last names (default 5000) |
| `gen_emails.py` | Generates `emails.txt` with guaranteed-unique addresses (default 200k) |
| `Makefile` | Builds all binaries, generates data, renders the PDF |

## References

1. Robey, Nicholaeff & Robey, *Hash-Based Algorithms for Discretized Data.*
2. Tumblin, Ahrens, Hartse & Robey, *Compact Hash Algorithms for Computational Meshes.*
3. Kao & Yoshimura, *Boosting GPU Radix Sort performance: A memory-efficient
   extension to Onesweep with circular buffers*, AMD GPUOpen, 2025 (*GPU Zen 3*).
   https://gpuopen.com/learn/boosting_gpu_radix_sort/ — source:
   GPUOpen-LibrariesAndSDKs/Orochi (`ParallelPrimitives`).
