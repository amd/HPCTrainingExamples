# Ghost Exchange in OpenMP offload: how many MPI ranks fit on one MI300A?

When several MPI ranks share a GPU they compete for the user-mode compute queues on the
`amdgpu` runlist. Exceed the budget and the driver time-slices the runlist: nothing
fails, everything gets slower, and the kernel log says why.

```
amdgpu: Runlist is getting oversubscribed due to too many queues. Expect reduced ROCm performance.
```

This is the OpenMP target offload half of the pair. Its sibling
[`../GhostExchange_Oversubscription_HIP`](../GhostExchange_Oversubscription_HIP)
measures the same thing for HIP and carries the shared background: where the queue
budget comes from, what changes in CPX mode, what MI250 looks like, and how Cray MPICH
on an HPE Cray EX differs. Read that one first. This one covers the single difference
that matters for OpenMP, which is that **a rank costs one queue more**.

Two names are used throughout: `NRANKS` is the number of MPI ranks placed on one
logical GPU, and `GPU_MAX_HW_QUEUES` is the environment variable that caps the hardware
queues a process may hold per device. It defaults to 4.

## The rule

A rank using OpenMP target offload holds `GPU_MAX_HW_QUEUES + 2` compute queues, where
the HIP equivalent holds `GPU_MAX_HW_QUEUES + 1`. The OpenMP runtime keeps a stream of
its own on top of the ones the application and the MPI library use. The budget is
unchanged: 24 compute queues per logical GPU on MI300A (gfx942), 21 on MI250 (gfx90a).

```
NRANKS x (GPU_MAX_HW_QUEUES + 2) <= budget    and, in CPX, NRANKS <= 6
```

| `GPU_MAX_HW_QUEUES` | queues per rank, HIP | ranks, HIP | queues per rank, OpenMP | ranks, OpenMP |
|---|---|---|---|---|
| 1 | 2 | 12 | 3 | 8 |
| 2 | 3 | 8 | 4 | 6 |
| 4 (default) | 5 | 4 | 6 | 4 |

So the two models agree at the default and diverge below it. If you are turning
`GPU_MAX_HW_QUEUES` down in order to pack more ranks onto a GPU — the main reason to
touch it at all — then using the HIP number for an OpenMP code will overshoot by a
third to a half.

## What is here

- `queue_probe_omp.cpp` counts the compute queues a rank owns at four points in its
  life, reading `/sys/class/kfd/kfd/proc/<pid>/queues` and using each queue's `type`
  file to separate compute queues from system direct memory access (SDMA) queues. It
  takes the number of target regions to run as its one argument.
- `sweep.sh` runs the OpenMP Ghost Exchange versions in
  `../GhostExchange_ArrayAssign` across a grid of `NRANKS` and `GPU_MAX_HW_QUEUES`
  values and writes a comma-separated file.
- `results_aac6_omp.csv` is the data behind the tables below, taken on an AMD MI300A
  under ROCm 7.2.2 with Open MPI 5.0.10.
- `mpirun_check.sh` is a launch wrapper that applies the rule to your command line
  before running it. It is byte-identical to the one in the HIP example and picks the
  right model from the executable, so there is nothing to choose.

```bash
module load rocm
module load openmpi
srun -p <mi300a-partition> -N1 -n1 --cpus-per-task=96 --gpus=4 --time=55 --pty bash -l

cd GhostExchange_Oversubscription_OpenMP
amdclang++ -fopenmp --offload-arch=gfx942 queue_probe_omp.cpp -o queue_probe_omp \
           $(mpicxx --showme:compile) $(mpicxx --showme:link)
GPU_MAX_HW_QUEUES=4 ROCR_VISIBLE_DEVICES=0 mpirun -n 2 --oversubscribe \
           --bind-to none ./queue_probe_omp 1

./sweep.sh -b -v "Ver1 Ver4 Ver6" -r "4 8 11 12" -q "1 2 4" -i 10000 -I 50
```

`ROCR_VISIBLE_DEVICES=0` pins every rank to one logical GPU, so adding ranks
oversubscribes a single device instead of spreading over the node. Build on the compute
node: `CMakeLists.txt` gets the target architecture from `rocminfo`, which finds no GPU
on a login node and yields a binary that dies at the first target region.

## Where the queues come from

`queue_probe_omp` with two ranks on one GPU. Columns are values of
`GPU_MAX_HW_QUEUES`:

| point in the run | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| first target region | 3 | 4 | 5 | 6 | 6 | 6 | 6 |
| after further target regions | 3 | 4 | 5 | 6 | 6 | 6 | 6 |
| after a GPU-aware `MPI_Sendrecv` | 3 | 4 | 5 | 6 | 6 | 6 | 6 |
| after a GPU-aware `MPI_Allreduce` | 3 | 4 | 5 | 6 | 7 | 8 | 10 |

The shape is the same as HIP's, shifted up by one. The count is lazy but arrives whole:
the very first target region already brings `GPU_MAX_HW_QUEUES + 2` queues, capped at 6
until a collective on a device buffer lifts the cap and takes it to
`GPU_MAX_HW_QUEUES + 2` outright. The larger number is the one to plan with, because
almost every real code does a device-buffer collective eventually.

How many target regions the application runs makes no difference: 1, 2 and 4 regions
gave identical counts at every `GPU_MAX_HW_QUEUES`. This is the same finding as on the
HIP side, and the same conclusion — the variable is a cap that the runtime and MPI fill
between them, not a count of what the application asked for.

**`LIBOMPTARGET_AMDGPU_NUM_HSA_QUEUES` is not the knob.** It looks like it should be,
and it is the name people reach for. Setting it to 1, 2, 4 and 8 left the count at 6
every time, which is just the `GPU_MAX_HW_QUEUES` default of 4 showing through.
`GPU_MAX_HW_QUEUES` is the variable that moves it, for OpenMP exactly as for HIP.

## What crossing the budget costs

Ghost Exchange `Ver1`, 10000x10000, 50 iterations, all ranks on one MI300A in SPX.
`Total` in seconds, with the predicted compute-queue total in parentheses:

| `NRANKS` | `GPU_MAX_HW_QUEUES`=1 | =2 | =4 |
|---|---|---|---|
| 4 | 0.478 (12) | 0.446 (16) | 0.457 (24) |
| 8 | 0.526 (24) | 1.086 (32) | 1.645 (48) |
| 11 | 1.430 (33) | 1.387 (44) | 1.440 (66) |
| 12 | 1.686 (36) | 1.918 (48) | 2.348 (72) |

Across all three versions and all twelve rank/queue combinations measured, every run
that fits in 24 queues took between 0.446 and 0.529 s and every run that does not took
between 0.926 and 2.645 s. There is no overlap between the two groups, and the cost is
entirely in the ghost exchange: the stencil time barely moves while `Ghost Cell Update`
goes from 0.01 s to over 2 s.

### The one-queue difference is visible in the application, not just the probe

The interesting cells are the ones where the two models disagree, because they predict
different breakdown points for the identical algorithm. Ghost Exchange `Ver1` in both
languages, same node, same problem:

| `NRANKS` | `GPU_MAX_HW_QUEUES` | HIP queues | HIP `Total` | OpenMP queues | OpenMP `Total` |
|---|---|---|---|---|---|
| 8 | 2 | 24 | 0.329 | 32 | 1.086 |
| 11 | 1 | 22 | 0.398 | 33 | 1.430 |
| 12 | 1 | 24 | 0.383 | 36 | 1.686 |

Each of these is a case where HIP sits at or under the budget and OpenMP is over it,
and in every one HIP stays fast while OpenMP is three to four times slower. The
boundary moves exactly where `+2` says it should, in both directions: 8 ranks at
`GPU_MAX_HW_QUEUES=1` is 24 queues for OpenMP and still fast at 0.526 s, while 8 ranks
at 2 is 32 and slow. That is the difference being confirmed end to end in a real code,
rather than inferred from the probe.

## Checking a job before you launch it

`mpirun_check.sh` reads the partition mode, the GPU family and your command line, works
out the queues, and tells you before anything runs. It detects OpenMP offload from
`libomptarget` in the executable's `NEEDED` entries and says which model it used:

```console
$ mpirun_check -n 4 --oversubscribe ./GhostExchange ...
mpirun_check: ok: 4 rank(s) over 4 logical GPU(s) = 1 per GPU, GPU_MAX_HW_QUEUES=4 -> up to 6 queues each, 6 of 24 (OpenMP offload) [gfx942 SPX]
```

Symlink or copy it under either name; the wrapped launcher follows the name.

```bash
ln -s mpirun_check.sh mpirun_check     # wraps mpirun
ln -s mpirun_check.sh srun_check       # wraps srun
```

`--recommended-settings` applies the fix instead of only reporting it, picking the
largest `GPU_MAX_HW_QUEUES` that keeps the job inside the budget. `--strict` refuses to
launch. The full option list is in `mpirun_check --help` and the rest of the behaviour,
including the Cray MPICH handling, is described in the HIP example's README.

### Which GPU does each rank actually use?

Every count above divides ranks by the number of visible GPUs, which is only right if
something assigns them. Nothing in OpenMP does by default, and
`GhostExchange_ArrayAssign` is a good illustration: `omp_set_default_device` is
*commented out* in all six versions, so every rank uses device 0 however many are
visible. The HIP example has the same hole by a different route, a bare
`hipSetDevice(0)`, and it is worth knowing that neither language protects you.

Unchecked this makes the wrapper's verdict not merely unhelpful but wrong. Twelve
ranks of `Ver6` on a four-GPU node, 4000x4000, 200 iterations:

| launch | reported | application `Total` |
|---|---|---|
| no shim | "ok: 3 per GPU, 18 of 24" | 12.918 s |
| `--set-affinity` | "ok: 3 per GPU, 18 of 24" | 0.200 s |

Same clean bill of health, 64x apart, because in the first row all twelve ranks are on
GPU 0 and the real figure is 72 queues of 24. Pass `--set-affinity` and the wrapper
gives each rank one GPU, round-robin by local rank, via `ROCR_VISIBLE_DEVICES` — which
steers OpenMP target offload just as it steers HIP, verified here with four ranks
landing on devices 0 through 3.

```console
$ mpirun_check --set-affinity -n 12 --oversubscribe ./GhostExchange ...
mpirun_check: ok: 12 rank(s) over 4 logical GPU(s) = 3 per GPU, GPU_MAX_HW_QUEUES=4 -> up to 6 queues each, 18 of 24 (OpenMP offload) [gfx942 SPX]
mpirun_check: setting affinity: one GPU each, round-robin over the 4 visible
```

Without it the wrapper names the right culprit for the language it detected:

```console
mpirun_check: note: this assumes the 12 rank(s) are spread over all 4 GPUs. A code with no omp_set_default_device
mpirun_check:   puts them all on one, which would be 72 queues of 24.
```

### Round-robin or block?

`--set-affinity` hands out devices round-robin by default, so neighbouring ranks land
on different GPUs; `--affinity-order=block` instead gives consecutive ranks the same
GPU. It matters, and OpenMP and HIP disagree about which way. Ghost Exchange `Ver6`,
8000x8000, 200 iterations, 4 GPUs, median of three, `Total` in seconds:

| ranks, grid | OpenMP round-robin | OpenMP block | HIP round-robin | HIP block |
|---|---|---|---|---|
| 12, 4x3 | **0.265** | 0.284 | 0.166 | **0.151** |
| 12, 3x4 | 0.257 | **0.226** | **0.110** | 0.156 |
| 12, 6x2 | 0.285 | **0.272** | **0.103** | 0.158 |
| 8, 2x4 | 0.226 | **0.193** | **0.091** | 0.132 |
| 16, 4x4 | 0.302 | **0.262** | 0.154 | 0.152 |

Block wins five of the seven configurations measured for OpenMP, by 5 to 15%, while
round-robin wins five of seven for HIP by as much as 53%. Both patterns reproduce, so
this is not noise, but neither is explained by counting how many neighbour pairs each
order puts on the same device — see the
[HIP README](../GhostExchange_Oversubscription_HIP) for that analysis and the
counterexample that breaks it. Treat the order as something to measure on your own
decomposition, not to reason about.

`--set-affinity=cpu` additionally pins each rank to the NUMA node its GPU sits on, and
`=both` does the two together. CPU pinning is off by default because it measured worth
nothing on this GPU-bound code and because it will override a binding the launcher or
Slurm has already set.

## What to do

- Budget `GPU_MAX_HW_QUEUES + 2` queues per rank for an OpenMP offload code, not the
  `+ 1` you would use for HIP, and keep `NRANKS x (GPU_MAX_HW_QUEUES + 2)` inside 24 on
  MI300A or 21 on MI250.
- At the default `GPU_MAX_HW_QUEUES=4` that means four ranks per logical GPU. To go
  beyond that, lower the variable rather than hoping: 2 buys six ranks, 1 buys eight.
- Reach for `GPU_MAX_HW_QUEUES`, not `LIBOMPTARGET_AMDGPU_NUM_HSA_QUEUES`.
- Try `--affinity-order=block`: it was the faster order for OpenMP in five of seven
  configurations measured, the opposite of what HIP preferred.
- Assign devices explicitly. Uncomment `omp_set_default_device`, or launch through
  `mpirun_check --set-affinity`, or use the `set_gpu_device.sh` shims in this tree.
  Without one of these every rank is on GPU 0 and none of the arithmetic applies.

## A note on the numbers

Everything here was measured on an AMD MI300A under ROCm 7.2.2. The queue counts were
identical under 6.4.3 on the HIP side, and the budget is a property of the `amdgpu`
runlist rather than of the ROCm release, so the arithmetic should travel. The timings
will not; they are one node, one problem size, and are here to show the shape of the
cliff rather than to be quoted.
