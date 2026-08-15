<!--
Copyright AMD 2026, MIT License
Author: Bob Robey Bob.Robey@amd.com with AI tool help
-->

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
prefix in a single pass. Concretely, each block first exposes its *local sum* and
later its *prefix sum*; a follower whose direct predecessor has not yet published a
prefix simply consumes that predecessor's local sum and keeps walking back, so no
block is held hostage by a single neighbor. The status flag is packed together with
the sum in one 32/64-bit word (updated atomically) and marked `volatile` so the
value is visible to other blocks immediately. This is a big part of why rocPRIM's
radix sort is fast.

The **Onesweep** algorithm (Adinets & Merrill) takes this further, fusing the scan
and reorder stages so the keys are read only once per pass. Onesweep is the current
fast-radix baseline, but it carries a subtle cost that is worth dwelling on: its
look-back needs a *temporal buffer whose size grows in proportion to n*. A recent
AMD result (Kao & Yoshimura, *GPU Zen 3*, 2025) removes that dependence with a
**constant-size circular buffer** — roughly 2 MB regardless of the number of keys —
governed by a "tail iterator" that marks which buffer slots are safe to overwrite
(with the look-back distance bounded below the buffer size). It is worth flagging
this now because it previews the paper's central theme: *even the general-purpose
radix sort runs into "memory scales with the wrong thing," and the fix is to
decouple the buffer size from the input.* The data-aware sorts below hit the same
wall against the key *range* and answer it the same way.

A clarification on where this lives: the circular-buffer variant is part of
**Orochi's `ParallelPrimitives`** (a GPUOpen library that loads HIP or CUDA at
runtime), *not* rocPRIM. What ships in ROCm — the `device_radix_sort` you get
through rocThrust or rocPRIM — is the standard Onesweep with decoupled look-back
(its look-back buffer still grows with n), continuously tuned for MI300 (for
example in ROCm 6.4 and again in ROCm 7.2 / rocPRIM 4.2.0). So the circular-buffer
optimization is a separate, buildable-from-source implementation, not a flag on the
stock library.

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

Notice the symmetry with the radix side. Onesweep's temporal buffer grew with the
number of keys and was tamed by a fixed-size circular buffer; the perfect-hash
table grows with the key *range* and, as we will see next, is tamed by the compact
hash. Same disease — memory tracking the wrong quantity — treated the same way, by
decoupling the allocation from the thing that blows up.

### Example: Unique ID sort -- a true perfect hash

The cleanest instance of a perfect hash is a set of keys that are both **dense**
and **unique**. Consider a mailing batch where every mailpiece carries a unique
sequential serial number (a USPS Intelligent Mail serial, say), so the keys form a
permutation of a contiguous range `[BASE, BASE+n)`. Two guarantees hold at once:
the range is dense (about `n` possible values for `n` keys) and each key occurs
exactly once (the mapping is a true bijection).

When both guarantees hold, sorting collapses to a single write per element with no
atomics, no histogram, and no scan:

```
slot[key[i] - BASE] = i;
```

Every destination slot is written exactly once, so there is no contention between
threads and nothing to reduce afterward — not even the compaction pass the general
spatial hash needs, because a dense unique range leaves no empty buckets to squeeze
out. This is the perfect hash in its purest form: one O(n) scatter and you are done.

The uniqueness guarantee is load-bearing, not incidental. If two keys collide, they
target the same slot: one write wins, another slot is never written, and a record is
silently lost — an outcome that a bijective input makes impossible but a violated
one makes invisible. That fragility is precisely what motivates the next case, where
the keys are *not* guaranteed unique and collisions must be handled explicitly.

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

## 7. Coming full circle: Orochi's circular-buffer radix sort

We opened with the Orochi result and can now close the loop, because it turns out
to be the *general-purpose* mirror image of the data-aware idea this paper has been
building toward. Recall the problem from Section 2: Onesweep is a fast single-pass
radix sort, but its decoupled look-back needs a temporal buffer whose size grows in
proportion to `n`. Sort a billion keys and the scratch memory grows with them, even
though at any instant only a bounded window of predecessors is actually being
consulted. The allocation tracks the *total input* when it only needs to track the
*active look-back distance*.

Kao & Yoshimura's fix (in Orochi's `ParallelPrimitives`) is to replace the linear,
n-sized buffer with a **fixed-size circular buffer** — on the order of 2 MB
regardless of key count. A "tail iterator" tracks the oldest slot still needed and
marks everything behind it as safe to overwrite, so blocks reuse a small ring of
slots instead of appending to an ever-growing array. Because the look-back distance
is bounded well below the buffer size, no block ever needs a slot that has already
been recycled, and correctness is preserved while the memory footprint becomes
**constant in `n`**.

That is precisely the move the compact hash makes, in a different guise:

- The **perfect spatial hash** allocates `range / Δmin` buckets — memory that tracks
  the *key range*, a quantity that can blow up (ZIP+4, adaptive-mesh refinement)
  even when the number of keys is modest. The **compact hash** decouples the table
  from the range by compressing wide hash codes into a table sized to the *number of
  entries*, absorbing the resulting collisions with quadratic probing.
- **Onesweep** allocates look-back scratch that tracks the *number of keys*. The
  **circular-buffer variant** decouples the buffer from `n` by recycling a small,
  bounded ring of slots, absorbing the bounded look-back distance within it.

Both are the same design pattern: *identify the quantity your memory is
accidentally tracking, prove that only a bounded slice of it is ever live at once,
and size the allocation to that bound instead.* The compact hash caps growth against
the key range; the Orochi sort caps growth against the input size. One is data-aware
and application-specific, the other is general-purpose and ships as a drop-in radix
sort — but they reinforce the same lesson, and seeing the radix baseline arrive at
it independently is the strongest evidence that "don't let memory scale with the
wrong thing" is a principle, not a trick.

There is a practical corollary. The circular-buffer sort is a separate,
buildable-from-source implementation in Orochi, not a flag on stock rocPRIM/
rocThrust (Section 2). If you are memory-bound on very large sorts — where the
Onesweep look-back buffer itself becomes a meaningful fraction of your budget — it
is worth reaching for, in the same spirit that you would reach for a compact hash
when a perfect-hash table would not fit.

## 8. Practical guidance

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
3. C.-C. Kao, A. Yoshimura. *Boosting GPU Radix Sort performance: A memory-efficient
   extension to Onesweep with circular buffers.* AMD GPUOpen, 2025 (also in *GPU Zen
   3: Advanced Rendering Techniques*). Source: GPUOpen-LibrariesAndSDKs/Orochi
   (`ParallelPrimitives`). `https://gpuopen.com/learn/boosting_gpu_radix_sort/`
4. A. Adinets, D. Merrill. *Onesweep: A Faster Least Significant Digit Radix Sort.*
   D. Merrill, M. Garland. *Single-pass Parallel Prefix Scan with Decoupled
   Look-back.*
