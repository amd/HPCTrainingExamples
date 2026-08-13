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

1. Sort a **nationwide mailer by ZIP code** with a perfect-hash (counting) sort.
2. Scale to **ZIP+4 / last names** where the key range is large or sparse, using a
   **compact hash**.
3. Run a **merge-free multi-node sort** (MPI + HIP) with distribution-aware
   partitioning.
4. **Exploit the MI300A APU's shared/unified memory** to eliminate host↔device
   copies (extra credit).
5. Profile everything (Appendix A).

> Build/PDF note: render with `pandoc hip_hash_sort_demo_MI300A.md -o demo.pdf`.

---

## 0. The MI300A APU: why "shared memory" changes the code

MI300A is an **APU**: the CPU (EPYC "Zen 4") cores and the CDNA3 GPU XCDs sit on
the same package and share **one physically unified pool of HBM3** with hardware
cache coherence. There is **no PCIe hop** and, importantly, **no separate device
copy of your data**.

Practical consequences for this exercise:

- A pointer from `hipMallocManaged` (or even ordinary `malloc`/`new` with XNACK on)
  is valid on **both** the CPU and the GPU. You generate data on the CPU and sort
  it on the GPU with **zero `hipMemcpy`**.
- Enable page migration / unified addressing:

```bash
export HSA_XNACK=1          # allow GPU to fault on host pages (unified memory)
```

- Query the arch (MI300A is `gfx942`) and compile for it:

```bash
rocminfo | grep -m1 gfx      # expect gfx942
export OFFLOAD_ARCH=gfx942
```

The **extra-credit** portions below show the copy-free pattern and mark where a
discrete-GPU code would have needed explicit transfers.

### Environment setup

```bash
module load rocm             # or: source /opt/rocm/env  (site-dependent)
hipcc --version              # confirm ROCm/clang toolchain
export HSA_XNACK=1
export OFFLOAD_ARCH=gfx942
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
openmpi` provides OpenMPI 5.0.10 and sets `$MPI_PATH`:

```bash
module load rocm/7.2.4 openmpi
make multinode_zip_sort MPI_HOME=$MPI_PATH
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

### 1.1 Starter code — `zip_sort.hip`

```cpp
// zip_sort.hip  —  perfect-hash / counting sort of records by 5-digit ZIP
// Build: hipcc -O3 --offload-arch=gfx942 zip_sort.hip -o zip_sort
#include <hip/hip_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define HIP_CHECK(x) do { hipError_t e=(x); if(e){ \
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); \
  exit(1);} } while(0)

static const int NBUCKETS = 100000;   // 00000..99999

// Histogram: count records per ZIP. (Exercise: try an LDS-privatized version.)
__global__ void histogram(const int* __restrict__ zip, int n, int* __restrict__ counts){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counts[zip[i]], 1);
}

// Scatter each record's id into its ZIP's slice using a running offset.
__global__ void scatter(const int* __restrict__ zip, int n,
                         int* __restrict__ offsets, int* __restrict__ out_id){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n){
        int pos = atomicAdd(&offsets[zip[i]], 1);
        out_id[pos] = i;                 // sorted order = ids grouped by ascending ZIP
    }
}

int main(int argc, char** argv){
    int n = (argc > 1) ? atoi(argv[1]) : (1<<20);   // start small, scale via argv

    // --- APU EXTRA CREDIT: unified memory, no hipMemcpy anywhere ---
    int *zip, *counts, *offsets, *out_id;
    HIP_CHECK(hipMallocManaged(&zip,     n*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&counts,  NBUCKETS*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&offsets, NBUCKETS*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&out_id,  n*sizeof(int)));

    // CPU fills the SAME physical memory the GPU will read (APU: zero-copy).
    srand(12345);
    for (int i=0;i<n;i++) zip[i] = rand() % NBUCKETS;
    for (int i=0;i<NBUCKETS;i++) counts[i]=0;

    // Optional on APU; on a discrete GPU you'd prefetch instead of copy.
    int dev=0; HIP_CHECK(hipGetDevice(&dev));
    (void)hipMemPrefetchAsync(zip, n*sizeof(int), dev); // best-effort locality hint

    int T=256, B=(n+T-1)/T;
    histogram<<<B,T>>>(zip, n, counts);
    HIP_CHECK(hipDeviceSynchronize());

    // Exclusive scan of counts -> offsets (start of each ZIP's slice).
    thrust::exclusive_scan(thrust::device, counts, counts+NBUCKETS, offsets);
    HIP_CHECK(hipDeviceSynchronize());

    scatter<<<B,T>>>(zip, n, offsets, out_id);
    HIP_CHECK(hipDeviceSynchronize());

    // CPU reads results directly — no copy back.
    bool ok=true;
    for (int i=1;i<n;i++) if (zip[out_id[i-1]] > zip[out_id[i]]){ ok=false; break; }
    printf("n=%d  sorted=%s  first ZIP=%05d  last ZIP=%05d\n",
           n, ok?"YES":"NO", zip[out_id[0]], zip[out_id[n-1]]);

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
the hood — the library workhorse), and **(C)** a compact hash (open addressing +
quadratic probing) built and verified over the large, sparse 64-bit name-key
space. Names are packed base-27 into a 64-bit key so numeric order equals
lexicographic order.

```cpp
// name_sort.hip  —  Example 2 reference solution (MI300A / gfx942)
// Build: hipcc -O3 --offload-arch=gfx942 name_sort.hip -o name_sort
// Run:   ./name_sort names.txt
#include <hip/hip_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

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
    HIP_CHECK(hipMallocManaged(&nbuf,(size_t)n*MAXLEN));
    HIP_CHECK(hipMallocManaged(&keys,(size_t)n*sizeof(uint64_t)));
    HIP_CHECK(hipMallocManaged(&keys2,(size_t)n*sizeof(uint64_t)));
    HIP_CHECK(hipMallocManaged(&letter,(size_t)n*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&id,(size_t)n*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&gcount,26*sizeof(int)));
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
    thrust::sort_by_key(thrust::device,keys2,keys2+n,id); HIP_CHECK(hipDeviceSynchronize());
    bool sorted=true; for(int i=1;i<n;++i) if(keys2[i-1]>keys2[i]){sorted=false;break;}
    printf("Alphabetical sort: %s  first=%s  last=%s\n", sorted?"OK":"FAILED",
           nbuf+(size_t)id[0]*MAXLEN, nbuf+(size_t)id[n-1]*MAXLEN);

    // Part C — compact hash build + verify
    int tableSize=(int)next_prime((long)(2*n+1));
    uint64_t* tkey; int* tcnt; unsigned long long* probe; int* notfound;
    HIP_CHECK(hipMallocManaged(&tkey,(size_t)tableSize*sizeof(uint64_t)));
    HIP_CHECK(hipMallocManaged(&tcnt,(size_t)tableSize*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&probe,sizeof(unsigned long long)));
    HIP_CHECK(hipMallocManaged(&notfound,sizeof(int)));
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

### 2.2 Exercises (extend the reference)

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
(local order + seams + load balance). Managed pointers are host-accessible on the
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

    int ndev = 1; (void)hipGetDeviceCount(&ndev);
    HIP_CHECK(hipSetDevice(rank % ndev));          // one APU per rank

    long n_local = (argc > 1) ? atol(argv[1]) : (1L<<22);
    int  skew    = (argc > 2) ? atoi(argv[2]) : 0; // 1 => skewed input

    int* zip; HIP_CHECK(hipMallocManaged(&zip, (size_t)n_local*sizeof(int)));
    srand(1000 + rank);
    for (long i = 0; i < n_local; ++i)
        zip[i] = skew ? ((rand()%100 < 80) ? (rand()%10000) : (rand()%NB)) : (rand()%NB);

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

    // 2) partition all local records by owner; MPI_Alltoallv
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
    int* recvbuf; HIP_CHECK(hipMallocManaged(&recvbuf,(size_t)(n_recv>0?n_recv:1)*sizeof(int)));
    MPI_Alltoallv(sendbuf.data(),sendcounts.data(),sdispls.data(),MPI_INT,
                  recvbuf,recvcounts.data(),rdispls.data(),MPI_INT,MPI_COMM_WORLD);

    // 3) local GPU counting sort of received records
    int *counts,*offsets,*out;
    HIP_CHECK(hipMallocManaged(&counts,NB*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&offsets,NB*sizeof(int)));
    HIP_CHECK(hipMallocManaged(&out,(size_t)(n_recv>0?n_recv:1)*sizeof(int)));
    for (int b=0;b<NB;++b) counts[b]=0;
    if (n_recv>0){ int T=256,B=(int)((n_recv+T-1)/T);
        histogram<<<B,T>>>(recvbuf,(int)n_recv,counts); HIP_CHECK(hipDeviceSynchronize());
        thrust::exclusive_scan(thrust::device,counts,counts+NB,offsets); HIP_CHECK(hipDeviceSynchronize());
        scatter_vals<<<B,T>>>(recvbuf,(int)n_recv,offsets,out); HIP_CHECK(hipDeviceSynchronize()); }
    bool local_ok=true; for(long i=1;i<n_recv;++i) if(out[i-1]>out[i]){local_ok=false;break;}

    // 4) merge-free: global offsets + seam check
    long gofs=0; MPI_Exscan(&n_recv,&gofs,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if (rank==0) gofs=0;
    int my_min=(n_recv>0)?out[0]:NB, my_max=(n_recv>0)?out[n_recv-1]:-1, prev_max=-1;
    if (rank>0)        MPI_Recv(&prev_max,1,MPI_INT,rank-1,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    if (rank<nranks-1) MPI_Send(&my_max, 1,MPI_INT,rank+1,7,MPI_COMM_WORLD);
    bool seam_ok=(rank==0)||(n_recv==0)||(prev_max<=my_min);

    long gmax=0,gsum=0; MPI_Reduce(&n_recv,&gmax,1,MPI_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_recv,&gsum,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    int lo=local_ok?1:0, so=seam_ok?1:0, glo,gso;
    MPI_Reduce(&lo,&glo,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&so,&gso,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
    printf("[rank %d] owns [%d,%d) recv=%ld goff=%ld local=%s seam=%s\n",
           rank,bnd[rank],bnd[rank+1],n_recv,gofs,local_ok?"YES":"NO",seam_ok?"YES":"NO");
    if (rank==0){ double bal=(gsum>0)?(double)gmax/((double)gsum/nranks):1.0;
        printf("GLOBAL: total=%ld imbalance(max/avg)=%.2f local=%s seams=%s (skew=%d)\n",
               gsum,bal,glo?"ALL-OK":"FAIL",gso?"ALL-OK":"FAIL",skew); }

    (void)hipFree(zip);(void)hipFree(recvbuf);(void)hipFree(counts);
    (void)hipFree(offsets);(void)hipFree(out);
    MPI_Finalize(); return 0;
}
```

### 3.3 SLURM launch on MI300A — `run_multinode.sbatch`

MI300A nodes expose 4 APUs. Launch one rank per APU with **`mpirun`, not `srun`**
(on this system the `srun --mpi=pmix` step launcher times out; `mpirun` is the
supported path). The batch script runs on the first allocated compute node, so
`mpirun` places the ranks from there. Each rank binds to device `rank % ndev` in
the code.

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

### 3.4 Exercises (extend the reference)

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

## 3.5 (Optional, advanced) The radix memory story: rocPRIM Onesweep vs. Orochi's circular buffer

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

**Part B — read the fix in Orochi (`ParallelPrimitives`).**
The GPUOpen circular-buffer extension (Kao & Yoshimura, *GPU Zen 3*, 2025) holds
that look-back buffer at a **constant ~2 MB regardless of `n`**. Clone Orochi and
read the mechanism — the `tail iterator`, `L_lookback`, and `N_table` in:

```bash
git clone --depth 1 https://github.com/GPUOpen-LibrariesAndSDKs/Orochi.git
$EDITOR Orochi/ParallelPrimitives/RadixSortKernels.h   # circular-buffer look-back
$EDITOR Orochi/ParallelPrimitives/RadixSort.cpp        # host driver + tail iterator
```

> **Build note (validated on this testbed).** Orochi is *not* a ROCm library; it is
> host-side driver-API code that loads HIP at runtime and compiles its kernels with
> hiprtc. It builds cleanly with plain `g++` (no `hipcc`), e.g.:
> ```bash
> g++ -O3 -std=c++17 -I. "-D__debugbreak()=abort()" -include cstdlib \
>     Orochi/Orochi.cpp Orochi/OrochiUtils.cpp \
>     contrib/hipew/src/hipew.cpp contrib/cuew/src/cuew.cpp \
>     ParallelPrimitives/RadixSort.cpp Test/RadixSort/main.cpp \
>     -ldl -lpthread -o build/RadixSortTest
> ```
> It links and initializes correctly on MI300A (`gfx942`) under both ROCm 7.2.4 and
> 6.3.0. **But the featured circular-buffer Onesweep path does not run correctly on
> MI300A**, and this is *not* a ROCm-version issue:
>
> | ROCm | small n (< 3072, single-pass kernel) | large n (Onesweep circular buffer) |
> |------|--------------------------------------|------------------------------------|
> | 7.2.4 | kernels don't launch (0.00 ms, fails) | fails |
> | 6.3.0 | **sorts correctly** (`n=1000` ✓) | **GPU memory fault / wrong result** (`n≥100k`) |
>
> Root cause: `ParallelPrimitives/RadixSortConfigs.h` hardcodes `WARP_SIZE = 32`
> and the reorder kernels divide the per-block work by 32. This is **correct on
> RDNA** — the AMD Radeon workstation/consumer GPUs run a **32-lane wavefront**, and
> Orochi's radix sort originates on that rendering/ray-tracing side, so it works as
> intended there. The **CDNA Instinct data-center GPUs (MI300A = CDNA3) use a
> 64-lane wavefront**, so the hardcoded 32 makes the multi-pass reorder index out of
> bounds. It is a **wave32-vs-wave64 portability gap, not a bug and not a ROCm
> version issue**: it would run on an RDNA workstation card but not on today's
> Instinct parts without a wave64 port (e.g. keying the constant off `warpSize` /
> `__AMDGCN_WAVEFRONT_SIZE__`). **Treat Part B as source study on MI300A**; the
> measurable takeaway is Part A.

**Exercises**
- **E3.5** Plot `temp_storage(n)` from Part A and fit the slope; confirm it is O(n).
  At what `n` does the scratch exceed your per-APU HBM budget alongside the data?
- **E3.6** In `RadixSortKernels.h`, identify where the circular buffer bounds the
  look-back distance (`L_lookback < N_table`) and explain, in two sentences, why
  that makes the temporal memory independent of `n`.

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
| `zip_sort.hip` | Example 1: perfect-hash / counting sort by ZIP (unified memory) |
| `name_sort.hip` | Example 2: LDS histogram + radix sort + compact hash (complete) |
| `multinode_zip_sort.hip` | Example 3: MPI range-partition + local sort (complete) |
| `rocprim_tempsize.hip` | §3.5: measures rocPRIM Onesweep temp-storage growth (validated) |
| `run_multinode.sbatch` | SLURM launcher (mpirun), 1 rank/APU on MI300A |
| `gen_names.py` | Generates `names.txt` with realistic surname-initial frequencies |
| `names.txt` | Sample attendee last names (default 5000) |
| `Makefile` | Builds all binaries, generates data, renders the PDF |

## References

1. Robey, Nicholaeff & Robey, *Hash-Based Algorithms for Discretized Data.*
2. Tumblin, Ahrens, Hartse & Robey, *Compact Hash Algorithms for Computational Meshes.*
3. Kao & Yoshimura, *Boosting GPU Radix Sort performance: A memory-efficient
   extension to Onesweep with circular buffers*, AMD GPUOpen, 2025 (*GPU Zen 3*).
   https://gpuopen.com/learn/boosting_gpu_radix_sort/ — source:
   GPUOpen-LibrariesAndSDKs/Orochi (`ParallelPrimitives`).
