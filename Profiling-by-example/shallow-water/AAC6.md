# Shallow-Water Profiling on AAC6

This page is the site guide for the
[Profiling-by-example shallow-water](../README.md) tutorials on AAC6.
The novice and advanced READMEs explain the profiling workflow and what each stage
teaches; here we cover only what is specific to AAC6 and the batch scripts provided
with the tutorial tree.

Hardware: MI300A on AAC6, ROCm `10.1.0a20260818` from the Ubuntu 24.04 nightlies
module tree.

## Repository layout

```
Profiling-by-example/shallow-water/
  env_aac6.sh                  # AAC6 template — copy to env.sh and edit
  env.sh                       # your local copy (not in git)
  setup_rocprof_compute_venv.sh
  setup_roofline_extractor_venv.sh
  submit.sh                    # sbatch wrapper (reads partition from env.sh)
  novice/0_baseline/ … 4_block_64x4/
    fom.sh                     # build, run, print MCUPS
    profile.sh                 # profiling and roofline commands for this stage
  advanced/0_baseline/ … 6_2d_decomposition/
    fom.sh
    profile.sh
    gpu_bind.sh                # per-rank GPU binding, two-NUMA nodes
    gpu_bind_cpx.sh            # per-rank GPU binding, single-NUMA nodes
```

Work through stages in order. Each stage directory builds on its own; the change
from one stage to the next is a few lines of source, documented in that stage's
README.

## One-time setup (login node)

Run these once after cloning or updating the repository.

```bash
cd Profiling-by-example/shallow-water
cp env_aac6.sh env.sh
# edit env.sh: set SLURM_PARTITION and, for advanced jobs, GPU_BIND / MPI_BIND
# optional: ROCPROFSYS_NETWORK_INTERFACE for NIC profiling (advanced stages 5–6)
./setup_rocprof_compute_venv.sh
export ROOFLINE_EXTRACTOR=/nfsapps/ubuntu-24.04/opt/rooflineExtractor
./setup_roofline_extractor_venv.sh
```

`env.sh` holds your local settings: Slurm partition, module versions, and
advanced-track MPI binding. It is gitignored so your settings stay local to
your setup.

For the advanced track, set `GPU_BIND` and `MPI_BIND` in `env.sh` to match
your node layout: `gpu_bind.sh` with `--map-by ppr:1:numa --bind-to numa`
on nodes with two NUMA domains, or `gpu_bind_cpx.sh` with `--map-by slot`
on single-NUMA nodes. The template comments show both pairs.

For NIC counter profiling in advanced stages 5–6, set `ROCPROFSYS_NETWORK_INTERFACE`
in `env.sh` to your node's HPC interface name after checking on a compute node:

```bash
rocprof-sys-avail -H -r net
```

NIC profiling blocks in the batch scripts are commented out until multi-node jobs
are available on AAC6; the variable can be set now so it is ready when those runs
are re-enabled.

`setup_rocprof_compute_venv.sh` creates `~/rocprof-compute-venv` and installs
the pinned Python packages that `rocprof-compute analyze` needs. Run it on a
login node where `pip` can reach the package index; compute nodes may not have
outbound network access.

Novice `profile.sh` scripts collect a roofline plot using whichever backend
`ROOFLINE_TOOL` selects in `env.sh`: `extractor` (default) runs
[`profile_app.py`](https://github.com/AMD-HPC/rooflineExtractor/blob/main/README.md);
`rocprof-compute` runs the `--roof-only` profile and analyze steps instead.

On AAC6, `env_aac6.sh` sets `ROOFLINE_EXTRACTOR` to the site install at
`/nfsapps/ubuntu-24.04/opt/rooflineExtractor` — that path supplies
`profile_app.py` and `requirements.txt`; no git clone is needed. Python
dependencies come from `~/roofline-venv` (created by
`setup_roofline_extractor_venv.sh` on a login node and activated in `env.sh`).
The `module load roofline-extractor/dev` line in `profile.sh` is a harmless
fallback when the venv is missing; it is not required once the venv exists.

On other systems, clone the repo, run `./setup_roofline_extractor_venv.sh`, and
set `ROOFLINE_EXTRACTOR` in `env.sh` (see the [novice README](novice/README.md#roofline-plots)).

If AAC6 moves to a different ROCm build, edit the `module load` lines in
`env.sh` to match.

### Validating all batch scripts

A separate validation harness can sync the tutorial tree to AAC6, submit every
`fom.sh` and `profile.sh` sequentially, wait for each job, print pass/fail, and
collect roofline plots. Expect several hours of queue time for the full run.

## Submitting jobs

Every stage has two scripts:

| Script | Purpose |
|---|---|
| `fom.sh` | Build with `make`, run the solver, print throughput (MCUPS) and correctness checks |
| `profile.sh` | Build, then run profiling commands from that stage's README (including roofline extractor on novice stages) |

Submit from inside the stage directory. Partition name comes from `env.sh`, not
from the `#SBATCH` lines (Slurm cannot expand shell variables there), so we use
`submit.sh`:

```bash
cd novice/0_baseline
../../submit.sh fom.sh
../../submit.sh profile.sh
```

Output lands in the stage directory as `fom_<jobid>.out` or
`profile_<jobid>.out`.

Check queue status:

```bash
squeue -u "$USER"
```

When the job has left the queue, read the result:

```bash
cat fom_*.out
# or
less profile_12345.out
```

## Novice track (one GPU)

Five stages, single-process HIP. Each `fom.sh` requests one GPU and 30 minutes;
each `profile.sh` requests one GPU and two hours (rocprof and roofline collection).

```bash
cd novice/0_baseline
../../submit.sh fom.sh        # expect ~6282 MCUPS (stage 0 baseline)
../../submit.sh profile.sh    # kernel trace, occupancy, roofline extractor
```

Then continue through `1_larger_domain`, `2_no_device_sync`, `3_block_32x32`, and
`4_block_64x4`. The stage READMEs list the expected MCUPS at each step.

## Advanced track (MPI, up to four GPUs)

Seven stages, multi-GPU MPI. Each `fom.sh` requests one exclusive node with four
GPUs and two hours. The script runs scaling loops at 1, 2, and 4 ranks using
`GPU_BIND` and `MPI_BIND` from `env.sh`. See the
[advanced README](advanced/README.md#affinity-which-decides-everything-else).

```bash
cd advanced/0_baseline
../../submit.sh fom.sh
../../submit.sh profile.sh
```

The `--exclusive` flag in the advanced batch scripts is load-bearing: without it,
ranks share a handful of CPU threads remote from the GPUs and timings are not
meaningful. Do not remove it.

Stage 6 `fom.sh` also builds stage 5 and compares slab vs tile decomposition.
Stage 5 and 6 `profile.sh` capture single-node `rocprof-sys` timelines at 4 ranks.
Stage 6 traces slab vs tile decomposition. Open Perfetto output at
[ui.perfetto.dev](https://ui.perfetto.dev).

NIC counter runs (`advanced/nic_trace.sh`, steps 10–29, two nodes) are commented
out in the batch scripts until multi-node jobs are available on AAC6. The stage
6 README has the full recipe; set `ROCPROFSYS_NETWORK_INTERFACE` in `env.sh`
after `rocprof-sys-avail -H -r net`.

## Things to keep in mind

**Login node vs compute node.** Module loads and `setup_rocprof_compute_venv.sh`
belong on the login node. Batch jobs load modules again via `env.sh` when they
start on a compute node.

**`env.sh` must exist before you submit.** Both `submit.sh` and every batch
script source `${SW_ROOT}/env.sh`. If you skip `cp env_aac6.sh env.sh` or leave
`SLURM_PARTITION` unset, submission fails immediately.

**`rocprof-compute analyze` and the venv.** Profile scripts call `analyze` after
`profile`. That requires the virtual environment from the one-time setup. If
`analyze` reports missing Python packages, re-run `setup_rocprof_compute_venv.sh`
on a login node.

**Roofline Extractor and Python.** Novice `profile.sh` scripts call
`profile_app.py` from `ROOFLINE_EXTRACTOR`. On AAC6 the install and Python deps
come from `~/roofline-venv` (see one-time setup); elsewhere use
`setup_roofline_extractor_venv.sh`.

**Counter and profile runtimes.** Profiling replays or multiplexes kernel
dispatches and can take much longer than a plain FOM run. Use `fom.sh` when you
only need throughput; reserve `profile.sh` for when you are collecting data.

**Output directories.** Profiling writes into the stage directory: `outdir/`,
`workloads/`, `trace/`, and similar paths listed in each track's `.gitignore`.
These can be large; remove them between stages if disk quota is tight.

**Correctness checks.** Every FOM run prints mass conservation and minimum depth.
If mass error jumps by orders of magnitude after an advanced-stage change, the
answer is wrong even if throughput improved. The stage 5 README shows an example
where a missing stream dependency passes a timer but fails physics.

**Numbers in the READMEs are reference points.** They were measured on MI300A in
SPX mode. Your absolute MCUPS will differ; the trends (occupancy rising with
domain size, communication dominating at small scale, and so on) are what matter.

## Quick reference: stages

| Track | Stage | Main idea |
|---|---|---|
| novice | `0_baseline` | Starting point; kernel trace and occupancy |
| novice | `1_larger_domain` | 512² → 2048² domain |
| novice | `2_no_device_sync` | Remove redundant `hipDeviceSynchronize()` |
| novice | `3_block_32x32` | 16×16 → 32×32 blocks |
| novice | `4_block_64x4` | 32×32 → 64×4 blocks |
| advanced | `0_baseline` | Scaling study; communication vs compute |
| advanced | `1_larger_domain` | 512² → 8192² global domain |
| advanced | `2_block_32x32` | Larger thread blocks |
| advanced | `3_block_64x4` | Wavefront-shaped blocks |
| advanced | `4_vectorized_loads` | Thread trace; explicit wide loads |
| advanced | `5_halo_pipeline` | Overlap halo exchange with compute |
| advanced | `6_2d_decomposition` | 2D tile vs 1D slab |

## Further reading

- [Novice track README](novice/README.md)
- [Advanced track README](advanced/README.md)
- [ROCm profiling guide, novice (blog)](https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/novice/README.html)
- [ROCm profiling guide, advanced (blog)](https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/advanced/README.html)
