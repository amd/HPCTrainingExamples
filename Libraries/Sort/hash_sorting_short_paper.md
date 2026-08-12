# Sorting on AMD GPUs: From General-Purpose Radix to Data-Aware Hash Sorts

*A short companion paper for the MPO Training Workshop, Sept 8–9, 2026.*

## 1. Why sorting still matters on the GPU

Sorting is one of the handful of primitives that shows up everywhere: reordering
records for output, grouping data for locality, building indices, preparing keys
for reductions and joins. On AMD GPUs the go-to implementations live in
**rocPRIM** (low-level device/block/warp primitives), **hipCUB** (a CUB-compatible
layer for easy CUDA ports), and **rocThrust** (a high-level, STL-like API). For
most workloads, calling `thrust::sort` or a rocPRIM device radix sort is the right
answer.

But "general purpose" has a cost. A general sort assumes nothing about the data,
and that assumption is exactly what we can exploit to go faster. This paper walks
through two data-aware techniques — the **perfect spatial hash sort** and the
**compact hash** — and shows how they extend naturally to a merge-free multi-node
sort. Two everyday examples carry the discussion: a nationwide mailer sorted by
ZIP code, and a conference registration table sorted by last name.

## 2. The baseline: radix sort

Radix sort is the fast general-purpose GPU sort. It is non-comparison based: it
processes keys a few bits (one "digit") at a time, and for each digit it builds a
histogram, runs a prefix scan, and scatters keys into their new positions. Each
pass is O(n), and the number of passes is set by the key width divided by the
number of radix bits per pass.

Modern implementations avoid the classic weakness — a separate global reduction
per pass — using a **decoupled look-back** scan, where thread blocks publish
partial sums and adaptively look back at their predecessors to compute their
prefix in a single pass. This is a big part of why rocPRIM's radix sort is fast.

The residual cost is structural: radix rereads the keys once per digit pass, and
that pass count is fixed regardless of how the data is actually distributed. If we
know something about the distribution, we can do less work.

## 3. The perfect spatial hash sort

The idea (Robey, Nicholaeff & Robey) is simple: if we know the data's minimum
value and the minimum spacing `Δmin` between distinct keys, we can map each key
*directly* to a bucket in a single pass:

```
bucket = floor((key - min) / Δmin)
```

Because the mapping is one-to-one (a *perfect* hash), there are no collisions and
no comparisons. Each thread writes its element (or its index) into a bucket with
no communication — an ideal fit for the GPU. The only real cost is a **prefix-scan
compaction** afterward to squeeze out the empty buckets and produce a dense,
sorted array. Where radix does a reduction per pass, the hash sort does one.

Reported results are striking on favorably distributed data: roughly **2–3×
faster than the GPU radix sort** and **~4× faster than a tuned CPU quicksort**.

**The trade-off is memory.** The hash table has `range / Δmin` buckets whether or
not they fill up. As the ratio of maximum-to-minimum spacing grows, the table
grows with it, and performance erodes. In practice the hash sort stays competitive
until that ratio reaches about 16 (equivalent to ~16 levels of adaptive mesh
refinement); beyond that, radix regains the lead. The lesson is not "hashing is
always faster" — it is "knowing your data is a performance lever."

### Example: the nationwide mailer

A five-digit ZIP code lives in a known, bounded range (00000–99999). That is the
textbook perfect-hash case: index directly by ZIP, one O(n) pass, about 100,000
buckets — small and dense. Sorting a mail run by ZIP becomes a single scatter plus
a compaction, no comparisons involved.

## 4. When the range explodes: the compact hash

Now switch the mailer to **ZIP+4** (nine digits, a billion possible codes) or to
full address strings. A perfect-hash table over a billion buckets is absurd when
you only have a few million labels. The **compact hash** (Tumblin, Ahrens, Hartse
& Robey) removes the memory ceiling by decoupling table size from key range.

Two ingredients make it work:

1. **A compression function** that maps the wide hash code into a small table while
   randomizing locations to break up patterns in the data. A simple 2-universal
   function suffices:

   ```
   hashLocation = ((a * hashCode + c) % m) % tableSize
   ```

   with linear-congruential-style constants. The table is now sized to the *number
   of entries* (its load factor), not to the range of possible keys.

2. **Collision handling**, because compression gives up the one-to-one guarantee.
   The paper uses **open addressing with quadratic probing**, `P(i) = A·i² + B·i`,
   over a prime (Proth-prime) table size so the first half of the probe sequence is
   guaranteed unique. Quadratic probing avoids the *primary clustering* that plagues
   linear probing. On the GPU, insertion is lock-free via atomics, and reducing the
   number of writes first (introducing sparsity) keeps load factors and probe counts
   low.

The cost is a few probes per collision; the payoff is memory bounded by the data
size rather than the key range — and the same O(n) target.

### Example: the conference registration table

Sorting attendee packets by last name looks trivial — there are only 26 letters —
but the distribution is heavily skewed (S, M, B, and C initials are common; Q, X,
and Z are rare). Two things fall out of this:

- If you bucket on the first *two or three* letters to get finer ordering, the key
  space is 676 or 17,576 slots, but a few hundred attendees fill almost none of it.
  A perfect-hash table would be mostly empty. The compact hash sizes to the
  attendees and lets probing absorb duplicate or near-identical names.
- The frequency skew is a load-balancing problem, which sets up the multi-node
  discussion below: registration desks (or ranks) should be assigned
  *frequency-weighted* letter ranges, not equal 26/N splits.

## 5. Scaling out: a merge-free multi-node sort

There is no turnkey AMD multi-node sort, but the hash approach composes into one
cleanly. Rather than "sort locally, then merge globally," we **range-partition up
front** — a distributed bucket sort:

1. Assign each rank a disjoint sub-range of the key space.
2. Hash each key once to its owning rank and shuffle it there (a single all-to-all
   over MPI or RCCL).
3. Sort locally on each rank (perfect or compact hash, or rocPRIM).

Because the ranges are disjoint and ordered, the global result is just the
**concatenation** of the rank outputs — the expensive distributed merge collapses
to a prefix-sum of per-rank counts. This is the same "one reduction, not many"
advantage the single-GPU hash sort enjoys, lifted to the cluster.

### Balancing the partition without a second read

Good range boundaries need an estimate of the global distribution. Sample sort
gets this by reading the data to sample it, then reading again to place it — two
touches of the data. We prefer a cheaper, single-pass scheme:

1. Start from an **assumed distribution** (uniform, or a known prior).
2. Place the **first ~10% of the data for real** under those boundaries — this is
   committed placement, not a throwaway sample.
3. **Adjust the boundaries** using what that 10% actually revealed.
4. **Move only the small misplaced fraction** of that early data, then stream the
   remaining 90% straight to the corrected boundaries.

The trade is deliberate: sampling pays a full extra read to mostly avoid data
movement; this method pays a small, bounded movement to avoid the extra read — the
better bet when re-reading and re-hashing the whole array is the expensive part and
a partial reshuffle over the interconnect is comparatively cheap. The failure mode
is instructive: skip the adjustment on skewed data (the conference last names, say)
and everything piles onto one rank.

## 6. A familiar pattern: MapReduce

Stepping back, the multi-node sort is exactly MapReduce with a distribution-aware
shuffle:

- **map** — hash each key to its range/rank,
- **shuffle** — the partition-driven all-to-all,
- **reduce** — the local (hash) sort.

The same pattern serves both examples: regional ZIP ranges for the mailer,
frequency-weighted letter ranges for the conference. Framing a GPU-native sort as
the shuffle-minimizing special case of a well-known pattern makes the design intent
easy to communicate.

## 7. Practical guidance

- Reach for **rocThrust** to get running, **rocPRIM** for control and peak
  performance, **hipCUB** for CUDA portability.
- Use a **perfect hash** when the key range is bounded and dense (five-digit ZIP).
- Use a **compact hash** when the range is large or sparse (ZIP+4, names on 2–3
  letters), accepting a few probes per collision in exchange for memory that scales
  with the data.
- For multi-node, **range-partition then concatenate**, and balance the partition
  with **assume–place–adjust** rather than a separate sampling pass.
- For related transforms, **rocFFT/hipFFT** cover single-device FFTs and **heFFTe**
  (rocFFT backend) covers distributed multi-node FFTs.

## References

1. R. N. Robey, D. Nicholaeff, R. W. Robey. *Hash-Based Algorithms for Discretized
   Data.*
2. R. Tumblin, P. Ahrens, S. Hartse, R. W. Robey. *Compact Hash Algorithms for
   Computational Meshes.*
