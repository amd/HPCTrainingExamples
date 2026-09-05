# Sorting on GPUs (HIP) — Library and Data-Aware Sorts

Hands-on sorting exercises for AMD GPUs (developed and tuned on the MI300A APU).
The progression starts with drop-in library sorts, then earns higher performance
with custom, data-aware hash sorts, and finally scales out to multiple GPUs/nodes.

The two companion write-ups contain the full walkthrough and results:

- [`sort_demo_MI300A.md`](sort_demo_MI300A.md) — the hands-on exercise guide (build,
  run, and analyze each sort; APU shared-memory notes; profiling appendix).
- [`hash_sorting_short_paper.md`](hash_sorting_short_paper.md) — background on the
  data-aware hash-sorting approach and when it beats a general-purpose primitive.

## What you will do

1. Start with the library baselines: `thrust` -> `rocPRIM` / `hipCUB` comparison-free
   radix sorts. These are the right default and the yardstick for everything else.
2. Perfect hash / direct-address sorts: when keys are unique on a bounded integer
   domain, each key maps to one slot in a single pass (calendar/date sort, ZIP/counting sort).
3. Collisions and compact hashing: when keys are sparse or non-unique (last names,
   emails), confront collisions and measure where an optimized hash does and does not help.
4. Index sort (argsort) vs. sorting the data in place.
5. Scale out: a merge-free multi-node sort (MPI + HIP) with distribution-aware partitioning.
6. APU shared memory: exploit MI300A unified memory to eliminate host<->device copies.

## Source files

- `sort_apis.hip` — library sort APIs side by side (thrust / rocPRIM / hipCUB, header-only)
- `index_sort.hip` — index sort (argsort) vs. sorting the data itself
- `date_sort.hip` — direct-address (calendar) sort on a bounded, gappy domain
- `zip_sort.hip` — counting / ZIP-code sort
- `name_sort.hip` — compact hash sort over last names (collisions)
- `email_sort.hip` — compact hash sort over sparse, unique email keys (load-factor / probe sweep)
- `multinode_zip_sort.hip` — merge-free multi-node sort (MPI + HIP)
- `tuned_multinode_zip_sort.hip` — data-aware, imbalance-triggered rebalance variant
- `rocprim_tempsize.hip` — measures rocPRIM Onesweep temporary-storage growth vs. n
- `gen_names.py`, `gen_emails.py` — sample data generators
- `run_multinode.sbatch` — example Slurm launch for the multi-node sorts
- `orochi_wave64.patch` — Orochi circular-buffer approach (reduced memory footprint)

## Build and run

```bash
make            # build all binaries + generate names.txt / emails.txt
make pdf        # render sort_demo_MI300A.md to PDF (needs pandoc + a LaTeX/tectonic engine)
make names.txt  # (re)generate the sample name list
make clean
```

The Makefile auto-detects the GPU arch via `rocminfo` and builds with `hipcc`.
On an APU, set `HSA_XNACK=1` to enable page migration for the shared-memory paths.
Multi-node targets use your MPI wrapper flags (`MPI_CXXFLAGS` / `MPI_LDFLAGS`).

See [`sort_demo_MI300A.md`](sort_demo_MI300A.md) for the step-by-step exercise and
the profiling appendix.
