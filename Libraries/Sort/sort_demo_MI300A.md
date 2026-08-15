---
title: "Hands-On: Sorting on MI300A (HIP)"
subtitle: "MPO Training Workshop — Sept 8, 2026"
author: "MPO Training"
date: "September 8, 2026"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

<!--
Copyright AMD 2026, MIT License
Author: Bob Robey Bob.Robey@amd.com with AI tool help
-->

# Sorting on MI300A — Hands-On Exercises

This exercise starts with simple sorting examples using existing libraries
for the AMD GPUs using HIP code. It then looks at custom sorting methods that
exploit data charistics for higher performance. Moving beyond a single MI300A
APU the examples show how to scale up to a multi-node run. The
progression is deliberate: start from the library sorts you already reach for,
then earn your way to custom hash sorts by understanding when — and why — knowing
your data lets you beat the general-purpose primitive.

The path we follow:

- **thrust → rocPRIM / hipCUB** — begin with the drop-in library baselines. These
  comparison-free radix sorts are the right default and the yardstick everything
  else is measured against, including index sort / argsort (§3.7).
- **perfect hash / direct address** — when keys are **unique** and land in a
  **bounded** integer domain, the key *is* the sort: each key maps to one slot, placed
  in one pass with no comparisons and no collisions. A fully dense domain is a *perfect
  hash*; a gappy one just compacts out the holes (calendar/date sort §3, counting/ZIP
  sort §4).
- **collisions and hashing** — when keys are **sparse or non-unique**, the perfect
  mapping breaks down. We introduce a **compact hash** and confront collisions
  head-on (last names §2, emails §2b), and mark where a highly optimized hash
  does and does not help.
- **scaling out** — a **merge-free multi-node sort** (MPI + HIP) with
  distribution-aware partitioning (§3).
- **alternative sorts** - Orochi's approach with circular buffer reducing memory requirements
- **APU shared memory** — exploit the MI300A's unified memory to eliminate
  host↔device copies (extra credit throughout).
- **profiling everything** — Appendix A.

> Build/PDF note: render with `pandoc sort_demo_MI300A.md -o demo.pdf`.

***

## 0. The MI300A APU: why "shared memory" changes the code

MI300A is an **APU**: the CPU (EPYC "Zen 4") cores and the CDNA3 GPU XCDs sit on
the same package and share **one physically unified pool of HBM3** with hardware
cache coherence. There is **no PCIe hop** and, importantly, **no separate device
copy of your data**. While there is no copy, there is still the cost for the cache
hierarchy to be filled separately for the CPU and the GPU. Setting `HSA_XNACK=1` is
needed to enable the memory pages to be migrated. 

While the MI300A is a true APU, most other data center GPUs support the APU 
programming model. They will just have the data transfer cost between the host
and device buffers that is done by the operating system.

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

> add `-p <slurm queue>` to request a particular queue

### Environment setup

- Enable page migration / unified addressing:
- Load ROCm and OpenMPI environments with module load
- The makefile queries the arch (MI300A is `gfx942`) and compiles for it:
- Checks for GPU and gpu type with rocminfo and hipcc toolchain

```bash
export HSA_XNACK=1           # allow GPU to fault on host pages (unified memory)
module load rocm openmpi
```

Checking that the environment is working

```bash
rocminfo | grep -m1 gfx
hipcc --version              # confirm ROCm/clang toolchain
```

***

## 1. Library sort APIs — thrust, rocPRIM, hipCUB (baseline & reference)

We begin with looking at the library routines. They are the easiest to implement
in an application. They will also establish a **baseline** performance. ROCm has
three ways to call library routines:

- **thrust** — the portable C++ STL-like API (`thrust::sort`, `thrust::sort_by_key`).
  On ROCm it dispatches to rocPRIM under the hood.
- **rocPRIM** — the AMD device-primitive library (`rocprim::radix_sort_keys`,
  `rocprim::radix_sort_pairs`). Lowest-level, most tuning knobs.
- **hipCUB** — a CUB-compatible wrapper over rocPRIM
  (`hipcub::DeviceRadixSort::SortKeys`/`SortPairs`). Use it when porting CUDA code
  that already calls CUB.

### 1.1 Using thrust -- the simplest approach

Thrust will allocate the temporary memory it needs and then perform the 
sort using rocPRIM. this makes it the easiest to use but least controllable.

The thrust sort call sorts the array ka in-place on the default device. It is easy to call
in your routine just by including thrust header files and with a single line of code. 

```bash
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
....
thrust::sort(thrust::device, ka, ka+n);
```

### 1.2 The two-call temporary-storage pattern (rocPRIM & hipCUB)

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
then again with `d_temp`). 

### 1.3 Side-by-side example — `sort_apis.hip`

`sort_apis.hip` sorts a **copy of the same array** with all three libraries
(keys-only, key-value pairs, and descending), verifies they agree bit-for-bit, and
prints hipEvent timings so you can compare them directly:

```bash
make sort_apis
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
the name baseline below. Since thrust calls the rocPRIM routine, the additional time
for the thrust can be assumed to be due to the time for the allocation. The sort time
itself should be the same.

### 1.4 Supported sort calls (quick reference)

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

### 2. Index sort (argsort) vs. sorting the data — `index_sort.hip`

Frequently you do **not** want to move the records — you want a *permutation* that
orders them. That is an **index sort** (argsort): sort an index array `[0,1,2,...]`
by the keys, leave the payload in place, then either read records indirectly or
**gather** once to materialize a sorted copy. `name_sort.hip` already does this
implicitly (`thrust::sort_by_key(keys, keys+n, id)` sorts `id[]`, not the names).

`index_sort.hip` makes the trade-off explicit and measured:

```bash
make index_sort
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

## 3. bounded-domain direct-address sort: unique keys on a dense-ish range

We start looking at examples where we can exploit unique characteristics of our
data array.

When keys are **unique** and land in a **bounded integer domain**, the key *is* its
own sorted position — no comparisons, no histogram, no scan. This is the
**direct-address** (bucket) idea in its cleanest form.

**Framing — a NY daily-newspaper archive.** We have `n` scanned issues, each stamped
with a **unique** publication date in the four-year window `2008-01-01 … 2011-12-31`.
"One issue per date" gives uniqueness for free, and the date becomes a bounded
integer index with a plain subtraction (0-based, C indexing):

```cpp
int day = date - 2008-01-01;   // 0 .. 1460   (DOMAIN = 1461 days; 2008 is a leap year)
slot[day] = i;                 // one write, no atomics: unique => one writer per slot
```

Two guarantees make this work:

1. a **bounded** integer domain, `day ∈ [0, DOMAIN)` with a `DOMAIN` we can afford to
   allocate, and
2. **unique** keys — each date occurs **exactly once** (one issue per day).

The scatter leaves the array in calendar order but **not packed**: a 4-year archive
of ~500 surviving issues fills only ~34% of the 1461 slots (missing issues, gaps in
the collection). So we add the one step the *fully dense* case never needs — **stream-
compact out the empty slots** — and the survivors emerge in sorted order:

```text
slot:  [ -1 , f7 , -1 , -1 , f3 , -1 , f9 , ... ]     scatter (calendar order, gappy)
        └──────────────── compact out the -1 ─────────┘
order: [ f7 , f3 , f9 , ... ]                          sorted issues, packed
```

> **The fully dense perfect hash is the special case.** Contiguous unique keys with
> `range ≈ n` — e.g. if we had a file for every day during that time period
> would fill the domain ~100%, so the compaction removes nothing and the scatter *is*
> the sort. But whether a real serial stream is gap-free is an **assumption**:
> The partially filled newspaper archive makes the gaps explicit instead of assuming them away.

### 3.1 Reference solution — `date_sort.hip`

`date_sort.hip` synthesizes `n` unique day-indices spread across the domain, shuffles
them (issues come off the scanner in random order), sorts them by **direct-address
scatter + stream compaction** (rocPRIM `select`), compares against a general library
radix sort of the same keys, and verifies the compacted order is strictly increasing
by date. Pass `dup=1` to give two issues the same date and watch a record get lost.

```bash
make date_sort
./date_sort                 # 500 issues over the 1461-day calendar (~34% full)
./date_sort 500 1461 1      # inject a duplicate date: uniqueness violated
./date_sort 1000000         # scales the domain to keep ~34% density
```

### 3.2 What it shows (measured on MI300A, gfx942, ROCm 7.2.4)

The direct-address sort does no comparisons and no atomics. Its cost is an `O(n)`
scatter **plus** `O(DOMAIN)` to clear and compact the calendar array — so what you
pay scales with the **domain**, not the input. Three runs (the domain scaled to hold
the same ~34% density) show where that helps and where it hurts:

```text
         n        domain   density   direct-address (scatter+compact)   library radix   ratio lib/direct
       500          1461     34.2%    0.112 ms (0.098 + 0.013)           0.011 ms        0.10
   1000000       2923976     34.2%    0.043 ms (0.022 + 0.022)           0.257 ms        5.91
 100000000     292397660     34.2%    4.964 ms (3.526 + 1.438)           2.739 ms        0.55
```

- **n = 500 (the actual newspaper archive):** both are microseconds and the numbers
  are dominated by kernel-launch latency, not the algorithm — effectively a tie. For
  data this small, just call the library sort.
- **n = 1M:** the direct-address sort is **~6× faster** — two cheap kernels (a
  scattered write, then a compaction) beat a multi-pass radix.
- **n = 100M:** with the domain scaled to ~292M slots (~1.2 GB), the `O(DOMAIN)`
  scattered write and compaction over a large *sparse* array now cost more than a
  bandwidth-optimized radix, and the **library wins**.

That is the density lesson in numbers: direct-address is unbeatable when the domain is
modest, but you pay for `O(DOMAIN)`, not `O(n)` — as the domain grows large and sparse,
the general radix (or the compact **hash** of §6) becomes the better tool.

Reference output for the default run and the duplicate-date case:

```text
# ./date_sort
date_sort: n=500 issues  domain=1461 days  density=34.2%  dup=0
  direct-address sort  : 0.112 ms  (scatter 0.098 + compact 0.013)  no atomics
  library radix sort   : 0.011 ms  temp=256B
  library/direct time ratio = 0.10  (>1 => direct-address faster)
  result: 500 issues in date order  (961 empty days compacted out)
    sorted[0]: file #183  ->  2008-01-02
    sorted[1]: file #424  ->  2008-01-03
    sorted[2]: file #109  ->  2008-01-06

# ./date_sort 500 1461 1   (duplicate date injected)
  result: NOT a clean bijection  selected=499 (expected 500)  order_ok=YES
  -> two files shared a date; one scatter overwrote the other, so a record was dropped.
```

The `dup=1` case is the whole lesson: two issues with the same date target the same
slot, one write wins, and a record is **silently lost**. That is precisely why the
next example, sorting by ZIP (non-unique), must fall back to a **counting sort** —
many records per bucket — instead of one-writer-per-slot placement.

### 3.3 Exercises

- **E3.1** Run `./date_sort 500 1461 1` and confirm exactly one issue is dropped
  (`selected=499`). Add a device-side duplicate detector (e.g. `atomicCAS` on a
  per-slot "written" flag) that reports which dates collided.
- **E3.2 (density sweep)** Fix `n` and grow the domain so the calendar gets sparser.
  The `O(n)` scatter stays roughly flat while the `O(DOMAIN)` compaction grows, so the
  `library/direct` ratio falls until the general radix wins. Measured at
  `n = 1,000,000` on MI300A (gfx942, ROCm 7.2.4):

```text
  ./date_sort 1000000 <domain>
        domain   density   direct-address (scatter+compact)   library radix   ratio lib/direct
       1000000    100.0%    0.136 ms (0.116 + 0.019)           0.257 ms        1.89
       3000000     33.3%    0.043 ms (0.022 + 0.021)           0.257 ms        6.05
      10000000     10.0%    0.059 ms (0.022 + 0.037)           0.258 ms        4.36
     100000000      1.0%    0.589 ms (0.029 + 0.560)           0.258 ms        0.44
    1000000000      0.1%    6.165 ms (0.050 + 6.115)           0.256 ms        0.04
```

  The crossover sits between ~10% and ~1% density: below it the compaction over a
  huge, mostly-empty array dominates and radix wins. The fully-dense (100%) row is the
  perfect-hash special case — compaction removes nothing, and the scatter alone still
  beats radix ~2×. Where does *your* crossover land, and how does it shift as you also
  grow `n`?
- **E3.3** Carry a payload: change `slot[day] = i` to also gather a record field, so
  the output is the fully reordered archive (not just the permutation of indices).
- **E3.4 (discussion)** When is a real key "dense enough" to direct-address? Compare
  the partial newspaper archive (bounded, ~34% full), a full newspaper archive (contiguous, ~100%),
  and an email address (unbounded string, effectively 0%). Where is the crossover to
  the compact hash of §6?

***
## 4. nationwide mailer, sort by ZIP (single APU)

Example 4 is honestly a **counting sort**, not a perfect hash: many addresses share
a ZIP, so keys are not unique and we need histogram → scan → **atomic** scatter. 

A 5-digit ZIP is a bounded, known range `[0, 100000)`. Because many addresses
share a ZIP, keys are **not unique**, so the "one index per bucket" perfect hash
generalizes to a **counting (bucket) sort**: histogram → exclusive-scan offsets →
scatter. This is a single O(n) pass plus one scan — still the pattern for the perfect hash.

### 4.1 Reference code — `zip_sort.hip`

The custom counting sort (histogram → `thrust::exclusive_scan` → scatter) runs on
unified memory, and the program then sorts a **copy** of the same keys with the
`rocprim::radix_sort_keys` **library baseline** so you get a fair custom-vs-library
comparison in one run (both timed *warm*; measured numbers in §3.7.5). See zip_sort.hip
for the source code.

### 4.2 Build & run (start small, then scale)

```bash
make zip_sort
./zip_sort 1024          # tiny, for correctness
./zip_sort 1000000       # 1M records
./zip_sort 100000000     # 100M — fits easily in MI300A unified HBM
```

### 3.7.5 Comparing the zip sort to library performance

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
  the multi-node partitioning in §7.

### 4.3 Exercises

- **E4.1** Compare `date_sort` and `zip_sort` at 1M / 100M. The direct-address sort
  avoids the scan and the atomic scatter — quantify how much each of those costs by
  disabling them in `zip_sort`.
- **E4.1** Confirm zero `hipMemcpy` calls (Appendix A shows how to verify with a
  trace). Compare against a version that uses separate `hipMalloc` +
  `hipMemcpy` and measure the difference on the APU. Or run the example on
  a discrete GPU that supports the APU programming model.
- **E4.2 (LDS extra credit)** ZIP has 100k buckets (too big for LDS), but the
  *scatter* is atomic-heavy. Try a **per-block LDS-privatized histogram** for the
  low 3 digits and combine — where does contention actually hurt?
- **E4.3** Make the sort carry a payload (weight/route) and produce a per-ZIP count
  report for the mail run.
- **E4.4 (baseline)** `zip_sort` prints a `rocprim::radix_sort_keys` time on
  the same keys. Compare it to the counting-sort time and explain the gap.

***


## 5. Conference last names (sparse keys → compact hash)

Encode the first `k` letters of a last name as a base-26 integer key. For `k=1`
there are 26 buckets (heavily skewed: S/M/B/C common; Q/X/Z rare). For `k=2,3` the
key space is 676 / 17,576 but a few hundred attendees fill almost none of it — a
hash table would be mostly empty. Use a **compact hash**: table sized to
the attendee count, collisions resolved by **open addressing with quadratic
probing**.

### 5.1 Reference solution — `name_sort.hip`

The complete program runs three parts on the same unified-memory data:
**(A)** an LDS-privatized first-letter histogram for registration-desk balancing,
**(B)** the actual alphabetical sort via `thrust::sort_by_key` (rocPRIM radix under
the hood — the library workhorse), plus a **library baseline** that calls
`rocprim::radix_sort_pairs` and `hipcub::DeviceRadixSort::SortPairs` directly on the
same keys for a head-to-head timing, and **(C)** a compact hash (open addressing +
quadratic probing) built and verified over the large, sparse 64-bit name-key space.
Names are packed base-27 into a 64-bit key so numeric order equals lexicographic
order.

> The three-library baseline during the run prints 
> `thrust::sort_by_key`, `rocprim::radix_sort_pairs`, and `hipcub::DeviceRadixSort`
> timings on the same 64-bit keys.

### 5.2 Build & run

```bash
make name_sort
./name_sort
```

### 5.3 Comparison of custom sort to library performance

- `name_sort` prints all three library timings on the same 64-bit name keys,
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

### 5.3 Exercises (extend the reference)

- **E5.1** The provided `names.txt` (from `gen_names.py`) matches real U.S.
  surname-initial frequencies. Run `./name_sort names.txt` and read off the
  frequency-weighted desk boundaries — do the four desks balance?
- **E5.2** Sweep the load factor: change `tableSize` to `1.1*n`, `1.5*n`, `2*n`,
  `4*n` and plot average probes/insert. Where does quadratic probing start to
  struggle?
- **E5.3** Replace quadratic probing with linear probing and observe primary
  clustering in the probe counts on the skewed name data.
- **E5.4 (discussion)** For `k=1` (26 keys) a dense counting sort wins; for the
  full 64-bit key the space is enormous and sparse so the compact hash is the
  memory-efficient structure. Explain the crossover in terms of load factor.
- **E5.5 (baseline)** The `name_sort` code prints `thrust::sort_by_key`,
  `rocprim::radix_sort_pairs`, and `hipcub::DeviceRadixSort` timings on the same
  64-bit keys. With the default 5000-name list these are microsecond-scale and
  launch-overhead-bound, so regenerate a large list first
  (`make names.txt NCOUNT=2000000`) to compare steady-state throughput.

***

## 6. emails: unique but *sparse* keys (compact hash)

This is the case people usually reach for when they hear "unique keys" — and it is
exactly why a perfect hash does **not** apply. An email is **unique** per recipient
(guarantee 2 from §3) but lives in a huge, sparse **string** space, so it fails the
**bounded-domain** guarantee 1. There is no `[0,n)` array to scatter into. Instead:

1. reduce each email string to a 64-bit key with a cheap device hash (FNV-1a), then
2. store the keys in a **compact hash** table sized `≈ n` (not `≈` the key range),
   resolving the collisions that the size-`n` compression creates by open addressing.

**FNV-1a** (Fowler–Noll–Vo, variant 1a) is a simple, fast, non-cryptographic
string hash. It folds a string into a 64-bit key one byte at a time — for each
byte it XORs the byte into the running hash, then multiplies by a fixed prime.
It needs no table lookups and just one multiply/XOR per byte, so it runs cheaply
on-device:

```cpp
// FNV-1a 64-bit string hash: cheap, good distribution, one multiply/xor per byte.
__host__ __device__ inline uint64_t fnv1a(const char* s, int maxlen){
    uint64_t h = 1469598103934665603ULL;   // FNV offset basis
    for (int j = 0; j < maxlen && s[j]; ++j){
        h ^= (uint64_t)(unsigned char)s[j];
        h *= 1099511628211ULL;             // FNV prime
    }
    return h;
}
```

The crucial lesson: **even with 100% unique emails you still get collisions after
compression**, so you still probe. Uniqueness buys you `distinct == n` (no lost
records), *not* a collision-free perfect hash.

### 6.1 Reference solution — `email_sort.hip`

`email_sort.hip` hashes each email to 64-bit (FNV-1a), builds the compact hash, and
**sweeps the load factor** to show probes vs. load. It also radix-sorts the hashes
as a bulk-dedup baseline (sorting by hash groups duplicates, but is *not*
alphabetical).

```bash
make emails.txt                 # gen_emails.py, 200k unique emails (ECOUNT to change)
make email_sort
./email_sort emails.txt
```

### 6.2 What it shows (validated on MI300A, ROCm 7.2.4)

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
  that a direct-address sort (`date_sort`, `date - base`) simply does not have.

### 6.3 Where "a highly optimized hash" fits (and where it does not)

- The **direct-address index** (§3) is `date - base` — a subtraction. It is
  **memory-bound on the scatter**, so a faster hash function changes nothing there.
- Here, in the **compact hash**, the *hash function* matters: a better-distributed
  hash lowers collisions/probes, and a cheaper one lowers per-key compute. FNV-1a is
  a fine cheap default; Murmur/xxHash-style finalizers are common alternatives.

### 6.4 Exercises

- **E6.1** Read off the probes-vs-load curve above. Which load factor is the best
  memory/probe trade-off for your query mix?
- **E6.2** Swap FNV-1a for a Murmur/xxHash-style 64-bit finalizer and compare
  `avg_probes/ins` and `distinct` (watch for any 64-bit collisions).
- **E6.3** Replace quadratic probing with linear probing and observe primary
  clustering at high load factors on the email keys.
- **E6.4 (APU)** Dedup the emails **on the CPU** with `std::unordered_map` (or
  `absl::flat_hash_map` if available) over the same unified buffer, and compare
  wall-clock and energy against the GPU compact hash. When is each the right tool?
- **E6.5** Instead of using a hash representation of the email address, is there
  another way that would allow using a **perfect hash** approach?

***

## 7. Scaling out — merge-free multi-node sort (MPI + HIP)

Range-partition the key space across ranks so the global result is a simple
**concatenation** (no distributed merge): each rank owns a disjoint ZIP range,
we move each record to its owner with a **neighbor-to-neighbor exchange** (no
all-to-all), then sort locally.

### 7.1 Distribution-aware partitioning (assume → place → adjust)

Rather than a separate sampling pass:

1. Start with **assumed** range boundaries (uniform split of `[0,100000)` across
   ranks).
2. Route the **first ~10%** of local records for real.
3. **Adjust** boundaries from the observed histogram of that 10%.
4. **Move** records with a neighbor-to-neighbor sweep (below).

### 7.1a Neighbor-to-neighbor redistribution (no all-to-all)

The data is assumed to be **written to the processor that owns that ZIP portion**
(pre-partitioned under the uniform assumption). Because the boundaries assign
**ascending** ZIP ranges to **ascending** ranks, a record that ends up on the
wrong rank belongs to one side, so the natural communication partner is the
**adjacent** rank in that order. Each round every rank:

1. Splits its buffer into *below-my-range* (→ left neighbor `rank-1`),
   *in-my-range* (keep), and *at/above-my-range* (→ right neighbor `rank+1`).
2. Exchanges those records with its two neighbors (`MPI_Sendrecv`; endpoints use
   `MPI_PROC_NULL`). Anything still out of range is **forwarded** on the next
   round, so records advance one hop per round toward their owner.
3. Stops when an `MPI_Allreduce` shows no rank moved anything. The returned
   `loops` count is the number of exchange rounds performed (`1` = a single
   neighbor communication).

When the guess is **close**, only the boundary stragglers move, so **one**
neighbor communication usually suffices and **two** is the safe bound. If the
guessed boundaries are **bad** — for example the 10% sample is unrepresentative
and collapses most of the data onto one processor — the sweep needs more than two
rounds; the code then **resets** the boundaries from a *full-data* histogram,
restores the original local records, and **re-places from scratch** (the
post-reset sweep is uncapped, so it always terminates in at most `nranks-1` hops
and never falls back to an all-to-all). Running with `skew=1` writes
badly-distributed input (80% of keys in the low band, not near their owner) to
exercise exactly this reset path.

### 7.2 Reference solution — `multinode_zip_sort.hip`

This example is a complete MPI + HIP program. It samples the first 10% to adjust boundaries,
redistributes with an iterative **neighbor-to-neighbor** exchange (`MPI_Sendrecv` to
`rank-1`/`rank+1`, with the two-loop-then-reset rule above — no `MPI_Alltoallv`),
sorts locally on the GPU, and validates the merge-free result
(local order + seams + load balance). hipMalloc pointers are host-accessible on the
APU, so the MPI calls need no separate staging buffers. Pass `skew=1` as the second
argument to force badly-distributed input and observe the reset/repair. Each rank
also reports `loops=` (neighbor-exchange rounds to converge) and `reset=` (whether
the full-histogram boundary reset fired). A **tuned variant**,
`tuned_multinode_zip_sort.hip`, keeps this same code and adds the recurring
**monthly-mailer** design case (a NE-heavy dataset with only ~10% churn that reuses
last month's boundaries and drives redistribution from the per-rank imbalance
instead of a fresh sample) — see §7.6.

### 7.3 Interactive SLURM run — single node, 4 APUs

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

We use a node-local bind logic where the program derives a **node-local rank** from an
`MPI_COMM_TYPE_SHARED` sub-communicator and calls `hipSetDevice(local_rank %
ndev)`, so on each node ranks bind to devices `0,1,2,3` (one APU each). It prints
its device and PCI-bus id at startup so you can confirm the mapping — e.g. on
one node the four APUs appear as `0000:01:00.0/.1/.2/.3`. This holds on 2 nodes ×
4 ranks too: each node runs its own shared-memory communicator, so the 4 ranks
per node get local ranks `0..3` regardless of the global rank numbering. An alternative is to use a
`ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` wrapper to set the APU (GPU) device
number. Because you stay on the allocated node, you can iterate quickly (rebuild
with `make multinode_zip_sort`, rerun `mpirun`) without resubmitting. Type `exit`
to release the allocation when you are done.

### 7.4 SLURM launch on MI300A — `run_multinode.sbatch`

MI300A nodes expose 4 APUs. Launch one rank per APU with **`mpirun`. The batch
script runs on the first allocated compute node, so `mpirun` places the ranks from there.

We use the same logic for setting the GPU devices for each MPI rank as was used
in the previous interactive example. 

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

The program warms up the GPU once (so `local_sort` reflects steady state, not
cold-start JIT/allocation), then reports five phase timers: `sample` (10% sample +
boundary adjust), `exchange` (neighbor sweeps + staging), `rebalance` (the
full-histogram reset overhead; `0` when no reset), `local_sort` (GPU counting
sort), and `dist_total` (barrier to seam check). `wasted_rounds` counts exchange
rounds thrown away before a reset.

```text
# PRE-PARTITIONED (skew=0): data already near its owner -> neighbor exchange only
[rank 0] owns ZIP [0,25581)     recv=3995451  ... loops=2 reset=0  |  sample=2.382  exchange=36.913  rebalance=0.000  local_sort=1.902  dist_total=62.119 ms
[rank 1] owns ZIP [25581,49997) recv=4003941  ... loops=2 reset=0  |  sample=2.380  exchange=37.558  rebalance=0.000  local_sort=1.769  dist_total=62.128 ms
[rank 2] owns ZIP [49997,74996) recv=3999788  ... loops=2 reset=0  |  sample=2.356  exchange=35.883  rebalance=0.000  local_sort=2.043  dist_total=62.200 ms
[rank 3] owns ZIP [74996,100000)recv=4000820  ... loops=2 reset=0  |  sample=2.345  exchange=43.587  rebalance=0.000  local_sort=1.989  dist_total=62.192 ms
GLOBAL: total=16000000  imbalance(max/avg)=1.00  local_sorted=ALL-OK  seams=ALL-OK  neighbor_loops=2 reset=NO wasted_rounds=0  (skew=0)
GLOBAL: max times (ms): sample=2.382  exchange=43.587  rebalance=0.000  local_sort=2.043  dist_total=62.200

# BADLY DISTRIBUTED (skew=1): 80% in the low band, not near owner -> reset path
[rank 0] owns ZIP [0,3047)      recv=4000276  ... loops=3 reset=1  |  sample=3.586  exchange=332.945  rebalance=7.550  local_sort=1.778  dist_total=358.401 ms
[rank 1] owns ZIP [3047,6099)   recv=4001222  ... loops=3 reset=1  |  sample=3.569  exchange=331.977  rebalance=7.523  local_sort=1.606  dist_total=358.431 ms
[rank 2] owns ZIP [6099,9147)   recv=4000716  ... loops=3 reset=1  |  sample=3.526  exchange=332.744  rebalance=7.388  local_sort=1.826  dist_total=358.423 ms
[rank 3] owns ZIP [9147,100000) recv=3997786  ... loops=3 reset=1  |  sample=3.550  exchange=325.667  rebalance=7.378  local_sort=1.981  dist_total=358.452 ms
GLOBAL: total=16000000  imbalance(max/avg)=1.00  local_sorted=ALL-OK  seams=ALL-OK  neighbor_loops=3 reset=YES wasted_rounds=2  (skew=1)
GLOBAL: max times (ms): sample=3.586  exchange=332.945  rebalance=7.550  local_sort=1.981  dist_total=358.452
```

With pre-partitioned input (skew=0) the guess is close, so only the boundary
stragglers move and the neighbor exchange converges in `loops=2` with **no reset**
(`exchange≈44 ms`, dominated by the two host-side partition passes over the local
buffer and the host→device staging, not the network). Under `skew=1` the data is
not near its owner: the guessed-boundary sweep exceeds two rounds
(`wasted_rounds=2`), the full-histogram **reset** fires (`rebalance≈7.5 ms`), and
the uncapped post-reset sweep re-places everything in `loops=3` (3 hops for 4
ranks) at `exchange≈333 ms`. The assume-place-adjust step still shrinks the
low-range partitions (rank 0 owns only `[0,3047)`) so per-rank counts stay
balanced (imbalance ≈ 1.00) and the disjoint ordered ranges make the global merge
a no-op. Note the steady-state `local_sort` is only ≈2 ms — the redistribution,
not the GPU sort, is what the neighbor-vs-reset choice governs. The gap between
`dist_total` and the phase timers (~15 ms) is the host-side local-order
verification and the seam exchange, which are intentionally left untimed.

### 7.5 Exercises (extend the reference)

- **E7.1** Run with `skew=0` then `skew=1`; compare the reported
  `imbalance(max/avg)`. Then **disable** the boundary adjustment (force uniform
  `bnd[]`) and rerun `skew=1` to reproduce the **degenerate all-on-one-rank** case.
- **E7.2** Log the number of records that changed owner after the 10% calibration
  vs. the uniform assumption (add a counter around the `dir_of` classification).
- **E7.3** Swap the ZIP key for a **last-name** key and reuse the compact hash from
  Example 2 as the per-rank local sort.
- **E7.4 (scaling study)** Run 1 → 2 → 4 → 8 nodes; report records/sec and the
  fraction of wall time in each phase timer (`sample`, `exchange`, `rebalance`,
  `local_sort`). How do `neighbor_loops`, `wasted_rounds`, and the number of hops
  grow with `nranks`, and when does the two-round budget stop being enough? How
  much of `exchange` is host-side partitioning/staging vs. actual MPI traffic?
- **E7.5** Try changing the amount of data used for the sampling from 10% to another
  value. What is optimal? How would this change for the amount of skew in the data?
- **E7.5a** With the default pre-partitioned input the sweep converges in `<=2`
  rounds with no reset. Increase the boundary-spill fraction (or the spill
  distance) in the generator until records must travel two ranks, and find the
  point where the two-round budget is exceeded and the reset fires. How does that
  threshold move as you change `nranks`?
- **E7.6** (discussion) When would the multinode approach be better than doing
  a single GPU sort?
- **E7.7 (monthly design case)** Build and run the tuned recurring-mailer variant
  (`tuned_multinode_zip_sort N 0 <churn%>`, see §7.6). With `churn=10` confirm
  `action=skip` (imbalance ≈ 1.02
  < `τ_nbr=1.05`) at both 4 and 8 ranks. Then raise the churn until the action
  becomes `neighbor` and then `reset`. Verify the crossover matches
  `τ_reset = 1 + 2/P` (1.50 at P=4, 1.25 at P=8) and that neighbor rounds track
  `loops ≈ (imbalance − 1) × P`.
- **E7.8 (projection)** From your E7.7 numbers, project the churn (or number of
  un-rebalanced months) at which a **16-** or **32-**rank job should reset rather
  than diffuse. Why does the reset threshold *tighten* as you add ranks?

***

## 7.6 Design case — the monthly recurring mailer (`tuned_multinode_zip_sort.hip`)

`skew=0/1` in `multinode_zip_sort` model a *cold* run: an unknown dataset sorted
from a uniform guess. Many real sorts are *recurring* — a monthly bulk mailing
re-sorts a database that is heavily northeast-weighted and changes only ~10% between
runs. The prior month already produced balanced boundaries and a histogram, so
re-sampling 10% of an almost-unchanged dataset is wasted work. The **tuned variant**
`tuned_multinode_zip_sort.hip` (built with `make tuned_multinode_zip_sort`) keeps
the original code intact and adds a third argument that turns on this mode:

```bash
./tuned_multinode_zip_sort <records/rank> 0 <churn%>   # e.g. 4000000 0 10  = 10% churn
```

In this mode the code (i) generates NE-heavy records already sitting under **last
month's** boundaries, (ii) applies `churn%` (half deletions, half NE-biased
additions), (iii) skips the 10% data sample and instead measures the **per-rank
count imbalance** `max/avg` — `P` numbers, one reduction — and (iv) applies a
tiered policy driven by two thresholds baked in near the top of the source:

```
TAU_NBR    = 1.05    // <= this: accept the drift, move nothing (action=skip)
NBR_BUDGET = 2       // neighbor rounds; tau_reset = 1 + NBR_BUDGET/nranks
```

- `imbalance ≤ τ_nbr (1.05)` → **skip**: no data moves; just the local sort + seam check.
- `τ_nbr < imbalance ≤ τ_reset (1 + 2/P)` → **neighbor** rebalance (budget ≈ `(imb−1)·P` rounds).
- `imbalance > τ_reset` → **reset**: rebuild boundaries in one shot and re-place.

Measured on one MI300A node (16M records):

```text
# 10% monthly churn — a typical month moves NOTHING
GLOBAL: total=16000001  imbalance_in=1.021 -> out=1.02  action=skip  ... neighbor_loops=0 reset=NO  (monthly churn=10%, tau_nbr=1.05, tau_reset=1.500)   # P=4
GLOBAL: total=16000002  imbalance_in=1.021 -> out=1.02  action=skip  ... neighbor_loops=0 reset=NO  (monthly churn=10%, tau_nbr=1.05, tau_reset=1.250)   # P=8

# heavy drift (proxy for many un-rebalanced months) — a bounded neighbor rebalance
GLOBAL: total=15999998  imbalance_in=1.207 -> out=1.00  action=neighbor  ... neighbor_loops=2 reset=NO  (monthly churn=100%, tau_nbr=1.05, tau_reset=1.250)  # P=8
```

The design numbers, from a P = 4/8/16 sweep:

| imbalance (in) | P=4 loops | P=8 loops | P=16 loops |
|---|---|---|---|
| 1.02 (10% churn) | 1 | 1 | 1 |
| 1.12 (50% churn) | 1 | 1 | 2 |
| 1.24 (100% churn) | 1 | 2 | 3 |
| 1.36 (150% churn) | 1 | 2 | 4 |

Neighbor rounds scale as **`loops ≈ (imbalance − 1) × P`**, so to stay within the
2-round budget the reset threshold is **`τ_reset = 1 + 2/P`** — it *tightens* as you
add ranks (thinner ranks ⇒ a given imbalance means mass must travel more hops, so a
one-shot rebuild beats a long diffusion). The takeaway: the generic path reads 10%
every run and budgets two neighbor rounds; the monthly design reads **0%** (a
maintained histogram + a `P`-value imbalance check), typically moves **nothing**,
and escalates to a neighbor rebalance or a machine-size-aware reset only when the
data actually drifts.

***

## 8. The radix memory story: rocPRIM Onesweep vs. Orochi's circular buffer

This alternative sort approach shows a memory trade-off versus the *general-purpose*
radix sort. Can we save memory use and still use the general sort algorithm?

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
make rocprim_tempsize
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

### 8.1 Examine the circular-buffer in Orochi and port it to wave64.
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

### 8.2 Porting to wave64

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
> compiles the kernels quickly. Earlier ROCm 6.x versions might not work.

### 8.3 Orochi sort exercises

- **E8.1** Plot `temp_storage(n)` from Part A and fit the slope; confirm it is O(n).
  At what `n` does the scratch exceed your per-APU HBM budget alongside the data?
- **E8.2** In `RadixSortKernels.h`, identify where the circular buffer bounds the
  look-back distance (`L_lookback < N_table`) and explain, in two sentences, why
  that makes the temporal memory independent of `n`.
- **E8.3 (port it yourself)** Starting from stock Orochi, reproduce the wave64 port:
  find every place a 32-lane assumption hides (the `WARP_SIZE` constant, the `/32`
  divisor, and the 32-bit `__ballot`/`__popc`/`__ffs`/`(1u<<lane)` in the reorder
  kernel) and fix them. Confirm correctness with the bundled test. This is the
  canonical "RDNA-tuned kernel → CDNA wave64" migration in ~30 lines.

***

### 9. Closing Exercises

- **9.1** Run `./sort_apis` at 1M, 16M, 64M keys. Do the three libraries agree?
  Which is fastest, and does the ranking change with `n`?
- **9.2** In `zip_sort`, compare the counting-sort time to the
  `rocprim::radix_sort_keys` baseline across `n` = 1M, 4M, 100M. At what `n` does the
  general radix sort overtake the counting sort, and why? (Hint: the scatter is
  atomic-bound into 100k HBM buckets.) When would the counting sort still be the
  right choice despite being slower here?
- **9.3** In `sort_apis`, add `begin_bit=0, end_bit=17` to the rocPRIM/hipCUB
  ZIP-range call (keys < 2^17) and measure the speedup from fewer radix passes.
- **9.4** In `index_sort`, sweep the payload from 1 to 64 words and plot
  index-sort+gather vs. data-sort time; identify the crossover.
- **9.5** Swap `name_sort`'s `thrust::sort_by_key` for the direct
  `rocprim::radix_sort_pairs` baseline and confirm identical alphabetical output.

***

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
- Correlate power dips with the neighbor-exchange redistribution phase (E3.4 timing).

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
sort, and (multi-node) the neighbor-exchange fraction and per-rank balance after
assume-place-adjust.

***

## Appendix B — File manifest

| File | Purpose |
|---|---|
| `zip_sort.hip` | Example 1: counting sort by ZIP (non-unique keys) + library baseline |
| `date_sort.hip` | Example 1b: direct-address (calendar) sort — unique keys on a bounded, gappy domain; scatter + compaction (+ dup detection) |
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
   `https://gpuopen.com/learn/boosting_gpu_radix_sort/` — source:
   GPUOpen-LibrariesAndSDKs/Orochi (`ParallelPrimitives`).
