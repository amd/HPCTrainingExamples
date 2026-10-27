# Ghost Exchange in HIP: how many MPI ranks fit on one MI300A?

When several MPI ranks share a GPU they compete for two driver resources that are
invisible from the application: the user-mode compute queues on the `amdgpu` runlist,
and the process slots the hardware scheduler can keep resident. Exceed either and the
driver starts time-slicing the runlist. Nothing fails; everything just gets slower, and
the kernel log says why:

```
amdgpu: Runlist is getting oversubscribed due to too many queues. Expect reduced ROCm performance.
amdgpu: Runlist is getting oversubscribed due to too many processes. Expect reduced ROCm performance.
```

This example measures both limits on an MI300A and shows what crossing them costs. It
uses the HIP Ghost Exchange, `GhostExchange_ArrayAssign_HIP`. Its sibling
`GhostExchange_Oversubscription_OpenMP` does the same for OpenMP target offload, where
a rank costs one queue more; the background below applies to both, so read this one
first.

Two names are used throughout: `NRANKS` is the number of MPI ranks placed on one
logical GPU, and `GPU_MAX_HW_QUEUES` is the HIP environment variable that caps the
hardware queues a process may hold per device. `GPU_MAX_HW_QUEUES` is the only spelling
the runtime recognizes; it defaults to 4.

## The rule

A rank holds `min(GPU_MAX_HW_QUEUES, streams it uses) + 1` compute queues, counting the
MPI library's streams along with the application's. Under Open MPI the library alone
uses enough of them to saturate the cap, measured at every `GPU_MAX_HW_QUEUES` from 1 to
8 and unchanged whether the application used 1 stream or 6, so in practice a rank costs
`GPU_MAX_HW_QUEUES + 1` and that is the number to plan with. The budget it is spent
against varies by GPU: MI300A (gfx942) allows 24 compute queues per logical GPU, MI250
(gfx90a) only 21.

```
NRANKS x (GPU_MAX_HW_QUEUES + 1) <= budget    and, in CPX, NRANKS <= 6
```

| `GPU_MAX_HW_QUEUES` | queues per rank | MI300A SPX | MI300A CPX | MI250 per GCD |
|---|---|---|---|---|
| 1 | 2 | 12 | 6 | 10 |
| 2 | 3 | 8 | 6 | 7 |
| 4 (default) | 5 | 4 | 4 | 4 |

MI300A in CPX mode carries a second limit of 6 processes per slice; in SPX the process
limit never fired in anything measured here, and on MI250 it cannot be reached without
exceeding the queue budget first. So CPX is the only case where the rank count itself,
rather than the queue total, is the thing to cap.

The default is the trap everywhere: at `GPU_MAX_HW_QUEUES=4` a fifth rank already
crosses the budget, and on MI300A the measured cost by eight ranks is a factor of
eight.

## What is here

- `queue_probe.cpp` counts the queues a rank owns at six points in its life, reading
  `/sys/class/kfd/kfd/proc/<pid>/queues` and using each queue's `type` file to separate
  compute queues from system direct memory access (SDMA) queues.
- `sweep.sh` runs the Ghost Exchange versions across a grid of `NRANKS` and
  `GPU_MAX_HW_QUEUES` values and writes a comma-separated file.
- `results_aac6.csv`, `results_aac6_cpx.csv` are the data behind the tables below,
  taken on an AMD MI300A under ROCm 7.2.2 with Open MPI 5.0.10 and UCX 1.19.1. The
  MI250 numbers use the same ROCm version.
- `results_aac6_order.csv` is the round-robin versus block data, three repeats of each
  cell, for both the HIP and the OpenMP build.
- `mpirun_check.sh` is a launch wrapper that applies the rule to your command line
  before running it. See "Checking a job before you launch it" below.

```bash
module load rocm
module load openmpi
srun -p <mi300a-partition> -N1 -n1 --cpus-per-task=48 --gpus=4 --time=55 --pty bash -l

cd GhostExchange_Oversubscription_HIP
hipcc --offload-arch=gfx942 $(mpicxx --showme:compile) queue_probe.cpp \
      -o queue_probe $(mpicxx --showme:link)
GPU_MAX_HW_QUEUES=4 ROCR_VISIBLE_DEVICES=0 mpirun -n 2 --bind-to none ./queue_probe

./sweep.sh -b -v "Ver1 Ver6 Ver8" -r "4 8 11 12" -q "1 2 4" -i 10000 -I 50
```

In `sweep.sh`, `-r` is the list of `NRANKS` values and `-q` the list of
`GPU_MAX_HW_QUEUES` values. `ROCR_VISIBLE_DEVICES=0` pins every rank to one logical
GPU, so adding ranks oversubscribes a single device instead of spreading over the node.
Build on the compute node: Ghost Exchange's `CMakeLists.txt` gets the target
architecture from `rocminfo`, which finds no GPU on a login node and yields a binary
that dies with `invalid device function`.

## Where the queues come from

`queue_probe` with two ranks on one GPU, identical on SPX and CPX, and identical again
under ROCm 6.4.3 and 7.2.2. Columns are values of `GPU_MAX_HW_QUEUES`:

| point in the run | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| before any HIP call | 0 | 0 | 0 | 0 |
| first kernel on the default stream | 2 | 2 | 2 | 2 |
| after `MPI_Init` | 2 | 3 | 5 | 6 |
| after a kernel on a second application stream | 2 | 3 | 5 | 6 |
| after a GPU-aware `MPI_Allreduce` | 2 | 3 | 5 | 9 |
| after a GPU-aware `MPI_Sendrecv` | +1 SDMA | +1 SDMA | +1 SDMA | +1 SDMA |

Three things to take from this. Queue creation is lazy, so a rank costs nothing until
its first kernel, which already brings two queues rather than one. The count then rises
to `min(GPU_MAX_HW_QUEUES + 1, 5)` after `MPI_Init` and holds there through the
application's own kernels, which makes point-to-point work look capped at five; a
device-buffer collective then takes it the rest of the way to `GPU_MAX_HW_QUEUES + 1`.
That is why the planning number is the larger one. Either way the variable is a cap
that the runtime and MPI fill between them, and the application's own stream count
barely enters into it: sweeping the application from 1 to 6 streams at each
`GPU_MAX_HW_QUEUES` from 1 to 8 moved the peak not at all. And the GPU-aware transfer
itself adds an SDMA queue, not a compute queue, so it is outside the 24-queue budget.

## SPX: the queue budget binds

Total seconds, 10000x10000 grid, 50 iterations, all ranks on one SPX GPU. Columns are
values of `GPU_MAX_HW_QUEUES` with the resulting queues per rank; bold exceeds the
24-queue budget.

| `NRANKS` | 1 (2/rank) | 2 (3/rank) | 4 (5/rank) |
|---|---|---|---|
| 4 | 0.149 | 0.144 | 0.147 |
| 8 | 0.174 | 0.162 | **2.251** |
| 11 | 0.217 | **0.466** | **1.414** |
| 12 | 0.198 | **1.345** | **3.654** |

Time tracks the queue total, not `NRANKS`: everything at or under 24 queues runs in
0.14 to 0.22 s whether that is 4 or 12 ranks, and everything past it is several times
slower. The two configurations landing on exactly 24 are both fast, so 24 is allowed and
25 is not. The cost lands on communication: in the worst case the halo exchange is 3.62
of the 3.65 s. Ver1 and Ver8 show the same shape; see
`results_aac6.csv`. The same sweep under ROCm 6.4.3 gives the same table to within a few
percent on every under-budget entry and puts the cliff in the same place.

## CPX: the process limit binds first

A CPX slice is one of six partitions of the physical GPU, with 38 compute units and its
own copy of both budgets. Ranks placed on one slice, 100 iterations, with
`GPU_MAX_HW_QUEUES=1` so queues are never the constraint:

| `NRANKS` | 2 | 4 | 6 | 7 | 8 | 12 |
|---|---|---|---|---|---|---|
| total (s) | 0.307 | 0.332 | 0.342 | 0.671 | 2.203 | 4.364 |

The seventh rank is where it breaks: six processes per CPX slice is free, the seventh
costs a factor of about two, and it keeps getting worse. The driver calls this one
"too many processes", not "too many queues". The boundary reproduces: repeating 4, 6 and
7 ranks with 30 s of idle on either side gave 0.316, 0.335 and 0.656 s, and the same
ladder under ROCm 6.4.3 gave 0.317, 0.333 and 0.803 s.

Reproduce either CPX table by running Ver6 directly at one rank count, since `sweep.sh`
varies `NRANKS` and `GPU_MAX_HW_QUEUES` together and so cannot separate the two limits.

The queue budget is still 24 per slice and the two limits are independent. Holding
`NRANKS` at 6, under the process limit, and raising `GPU_MAX_HW_QUEUES` instead:

| `GPU_MAX_HW_QUEUES` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| queues | 12 | 18 | 30 | 36 |
| total (s) | 0.328 | 0.319 | 1.415 | 1.587 |

So on CPX you must satisfy both: at most 6 ranks per slice, and at most 24 queues. At
the default `GPU_MAX_HW_QUEUES=4` the queue budget is the tighter of the two and allows
only 4.

## MI250: same formula, smaller budget

MI250 is a two-die card, so each of its 8 logical GPUs is one graphics compute die
(GCD) with 104 compute units. `queue_probe` gives exactly the same per-rank counts
there as on MI300A, 2, 3, 5 and 6 for `GPU_MAX_HW_QUEUES` of 1, 2, 4 and 8, so the
formula carries over unchanged. The budget does not.

Ver6 on one GCD, 100 iterations, `GPU_MAX_HW_QUEUES=1` except where noted, with the
kernel log attributed per run:

| `NRANKS` | 4 (at 4) | 7 (at 2) | 10 | 11 | 12 | 13 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| queues | 20 | 21 | 20 | 22 | 24 | 26 | 28 | 32 |
| total (s) | 0.220 | 0.264 | 0.353 | 0.717 | 286 | 4.06 | 9.31 | 13.7 |
| driver message | none | none | none | queues | queues | queues | processes | processes |

21 queues is clean and 22 is not, so MI250's budget is 21 rather than MI300A's 24. Two
further things are worth knowing. The 12-rank case is pathological rather than merely
slow, 286 s against 0.353 s for 10 ranks, and it reproduced on a repeat run (280 s the
first time), so it is not a one-off. And the process trigger does appear on MI250, at
14 ranks, but only in runs that are already far over the queue budget, so in practice
the queue rule is the only one you need there.

This is why the wrapper keys its budgets off `gfx_target_version` from the kernel
fusion driver topology rather than assuming one number: 90402 for gfx942 and 90010 for
gfx90a. On an unrecognized family it says so and falls back to 24.

## Checking a job before you launch it

`mpirun_check.sh` wraps the launcher, works out `NRANKS` per GPU from the command line
and the budgets from the node, and says something only when there is something to say.
Install it under either name; the launcher it wraps follows the name it is called by.

```bash
ln -s mpirun_check.sh mpirun_check     # wraps mpirun
ln -s mpirun_check.sh srun_check       # wraps srun
```

Then put the wrapper where the launcher used to be:

```console
$ mpirun_check -n 12 --map-by core:OVERSUBSCRIBE ./GhostExchange -x 4 -y 3 ...
mpirun_check: WARNING: the amdgpu runlist will be oversubscribed and every rank slows down.
mpirun_check:   12 rank(s) over 1 logical GPU(s) = 12 per GPU, GPU_MAX_HW_QUEUES=4 -> 5 queues each, 60 of 24 [gfx942 SPX]
mpirun_check:   over the 24-queue budget: expect "Runlist is getting oversubscribed due to too many queues" in dmesg
mpirun_check:   fix: GPU_MAX_HW_QUEUES=1 keeps this rank count inside the budget
mpirun_check:   or re-run with --recommended-settings to apply it
```

It warns and launches anyway, so dropping it into an existing script cannot break a
working job. Three flags change that: `--recommended-settings` sets
`GPU_MAX_HW_QUEUES` to the largest value that fits before launching, `--strict` refuses
to launch an over-budget job, and `--quiet` suppresses the confirmation line when the
configuration is already fine. On the command above, adding `--recommended-settings`
took the run from 3.106 s to 0.183 s without any other change.

Two more, `--set-affinity` and `--affinity-order`, decide which GPU each rank gets;
they are the subject of the next two sections. `mpirun_check --help` lists them all.

### Which GPU does each rank actually use?

Every count above divides ranks by the number of visible GPUs, which is only right if
something assigns them. Nothing in HIP does. `Ver6/GhostExchange.hip` calls
`hipSetDevice(0)`, like plenty of real codes, so all its ranks land on GPU 0 however
many are visible, which is why the examples in this tree are always launched through a
`set_gpu_device.sh` shim.

Forget the shim and the check is not just unhelpful, it is wrong. Twelve ranks on a
four-GPU node, measured on AAC6:

| launch | reported | application `Total` |
|---|---|---|
| no shim | "ok: 3 per GPU, 15 of 24" | 15.2 s |
| `--set-affinity` | "ok: 3 per GPU, 15 of 24" | 0.128 s |

Same clean bill of health, 121x apart, because in the first row all twelve ranks are
on GPU 0 and the real figure is 60 queues of 24.

So the wrapper now says which assumption it is making, and `--set-affinity` makes that
assumption true by giving each rank one GPU, round-robin by local rank:

```console
$ mpirun_check -n 12 --oversubscribe ./GhostExchange ...
mpirun_check: ok: 12 rank(s) over 4 logical GPU(s) = 3 per GPU, ... 15 of 24 [gfx942 SPX]
mpirun_check: note: this assumes the 12 rank(s) are spread over all 4 GPUs. A code that calls
mpirun_check:   hipSetDevice(0) puts them all on one, which would be 60 queues of 24.
mpirun_check:   pass --set-affinity to make the assumption true.
```

It reads `OMPI_COMM_WORLD_LOCAL_RANK`, `SLURM_LOCALID` or `MPI_LOCALRANKID`, whichever
the launcher sets, so it works under both `mpirun` and `srun`. On AAC7 under Cray MPICH
the same twelve ranks go from 1.143 s to 0.659 s.

Two limits worth knowing. It only acts when the executable appears as a path in the
command line, since that is how it finds where to insert itself; wrap your own script
and it will say so and launch unchanged. And it assigns round-robin, which is right for
the common case but not for a code that wants a particular rank-to-GPU order.

CPU pinning is a separate value, `--set-affinity=cpu` or `=both`, and off by default
on purpose. Each of the four MI300A dies is its own NUMA node of 24 cores and GPU *k*
really does sit on node *k* (read from `/sys/class/drm/renderD*/device/numa_node`
rather than assumed), so it looks like it should matter. It did not: `Total` 0.057 s
with `--bind-to numa --map-by numa` against 0.058 s without, across three repeats, and
deliberately putting rank *k* on GPU 3−*k* cost only about 8%. That is one GPU-bound
code, so the flag is there for host-heavy ones, but it will fight a binding the
launcher or Slurm has already set, which is why you have to ask for it.

### Cray MPICH

`srun_check` is worth using on a Cray system too, and the reason is worth understanding
because it is where the rule above comes from.

The same law applies: a rank holds `min(GPU_MAX_HW_QUEUES, streams it uses) + 1`
queues. What differs is who supplies the streams. Cray MPICH contributes exactly one of
its own, created by the GPU transport layer for device-buffer collectives, where Open
MPI contributes enough to saturate the cap by itself. So under Cray MPICH the
application's own stream count does matter, and the peak is
`min(GPU_MAX_HW_QUEUES, app streams + 1) + 1`.

Measured on an HPE Cray EX MI300A by sweeping both knobs, peak compute queues per rank:

| app streams | `GPU_MAX_HW_QUEUES` 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 3 | 3 | 3 | 3 | 3 |
| 2 | 2 | 3 | 4 | 4 | 4 | 4 | 4 |
| 3 | 2 | 3 | 4 | 5 | 5 | 5 | 5 |
| 4 | 2 | 3 | 4 | 5 | 6 | 6 | 6 |
| 6 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |

Read down the diagonal: the cost is whichever runs out first, the variable or the
streams. A single-stream application at the default `GPU_MAX_HW_QUEUES=4` costs 3
queues and fits 8 ranks on a 24-queue GPU; a 3-stream application at the same setting
costs 5 and fits only 4, exactly like Open MPI. Running the identical sweep under
Open MPI on AAC6 gives `GPU_MAX_HW_QUEUES + 1` in every cell, 1 stream or 6.

The wrapper cannot know how many streams your application uses, so it plans with the
worst case, `GPU_MAX_HW_QUEUES + 1`, and tells you when Cray MPICH might let you off
more cheaply:

```console
$ srun_check -n 8 ./my_cray_app
mpirun_check: WARNING: the amdgpu runlist will be oversubscribed and every rank slows down.
mpirun_check:   8 rank(s) over 1 logical GPU(s) = 8 per GPU, GPU_MAX_HW_QUEUES=4 -> up to 5 queues each, 40 of 24 [gfx942 SPX]
mpirun_check:   fix: GPU_MAX_HW_QUEUES=2 keeps this rank count inside the budget
mpirun_check:   note: under Cray MPICH only one of those queues is MPI's, so an application
mpirun_check:         using fewer than 4 streams costs less than 5 and may already fit
```

It identifies the stack from the application's `NEEDED` entries, since Cray MPICH links
`libmpi_cray` and `libmpi_gtl_hsa` where Open MPI links `libmpi.so`. A loaded
`cray-mpich` module proves nothing about what the binary was linked against, so that is
only the fallback when the executable cannot be found in the command line.

### Measured on an HPE Cray EX, end to end

The table above is queue counting. Running GhostExchange Ver6 itself on AAC7 (MI300A,
SPX, Cray MPICH, `srun`, one GPU, timings are the application's own `Total:`) shows
what it costs, and which driver message fires at each rank count:

| ranks | total (s) | driver messages during that run |
|---|---|---|
| 8 | 0.53 | none |
| 10 | 0.54 | 2x "No more SDMA queue to allocate" |
| 11 | 0.57 | 3x SDMA |
| 12 | 0.54 | 4x SDMA |
| 13 | hung | 5x SDMA, **"too many queues"** |
| 14 | 0.96 | 6x SDMA, "too many processes" |
| 16 | 1.27 | 8x SDMA, "too many processes" |

Three things worth separating here.

The compute-queue message appears for the first time at 13 ranks, which is exactly
where the model predicts it: this code uses one stream and never calls a device-buffer
collective, so the transport layer's stream is never created and each rank holds 2
queues. Twelve ranks is 24, precisely the budget; thirteen is 26 and over. That is also
why `GPU_MAX_HW_QUEUES` makes no difference anywhere in this sweep, which was verified
separately at 4, 2 and 1 across the same rank counts.

The SDMA messages are a *different resource* and start much earlier, at 9 ranks. There
are 16 SDMA queues per GPU and a rank takes 2, so 8 ranks can use the fast copy path
and every rank beyond that logs "No more SDMA queue to allocate" and falls back. The
count of failing ranks is exactly `ranks - 8` at every point in the table. It is a soft
limit: 12 ranks still runs at full speed.

The 13-rank hang reproduced across two separate jobs and remains unexplained; 14 and 16
ranks run normally. It is the same shape as a 12-rank anomaly seen on MI250 and should
not be read as the general cost of crossing the queue budget.

Note how much gentler this is than Open MPI: 16 ranks on one GPU costs 2.4x here, where
12 ranks under Open MPI on AAC6 cost 45x. The queue budget is the same 24 on both
machines; the difference is entirely in how many streams the MPI library adds.

### Using it on a Cray system

`srun_check` is the same script under another name, and it runs on AAC7 unchanged:

```console
$ srun_check --ntasks=12 --overlap ./GhostExchange -x 12 -y 1 ...
mpirun_check: note: 12 rank(s) over 1 logical GPU(s) = 12 per GPU, 24 to 60 queues of 24 [gfx942 SPX]
mpirun_check:   under Cray MPICH the cost depends on how many streams the code uses: 2 per rank
mpirun_check:   for a single-stream code, which fits 12, rising to 5 per rank for several
mpirun_check: note: past 8 rank(s) per GPU the 16 SDMA queues run out, so 4 rank(s) will
mpirun_check:   log "No more SDMA queue to allocate" and fall back to a slower copy path
```

Because it cannot know the application's stream count it reports the range rather than
a single number, and only escalates to a warning when even the 2-per-rank best case is
over budget. At 16 ranks it does: 32 of 24, which is the run that actually slowed down.

What carries over completely unchanged is the CPX process limit. Six processes per
slice is a property of the driver, not of the MPI library, so it bites at the seventh
rank however cheap that rank is in queues. On a partitioned Cray system that is the
limit you are most likely to hit.

### Turning GPU-aware MPI on in the first place

Cray MPICH needs two separate things to accept a device pointer, and neither is the
default:

1. `libmpi_gtl_hsa`, the GPU transport layer, linked into the executable. The compiler
   wrappers add it only when a `craype-accel-amd-gfx*` module is loaded.
2. `MPICH_GPU_SUPPORT_ENABLED=1` in the environment at run time.

Missing either one fails in a different way, so both are worth knowing:

| GPU transport layer | `MPICH_GPU_SUPPORT_ENABLED` | what happens |
| --- | --- | --- |
| linked | `1` | GPU-aware MPI works |
| linked | unset | runs, but device pointers are not supported |
| not linked | unset | runs, but device pointers are not supported |
| not linked | `1` | `MPI_Init` aborts: "GPU_SUPPORT_ENABLED is requested, but GTL library is not linked" |

The silent middle rows are the dangerous ones. The wrapper checks both: it reads the
executable's `NEEDED` entries for the transport layer and the environment for the
variable, and `--recommended-settings` exports the variable when the transport layer is
there to back it up.

```console
$ srun_check -n 4 ./my_cray_app
mpirun_check: ok: 4 rank(s) over 1 logical GPU(s) = 4 per GPU, 3 queues each under Cray MPICH, 12 of 24 [gfx942 SPX]
mpirun_check: WARNING: MPICH_GPU_SUPPORT_ENABLED is not set, so Cray MPICH will not accept device pointers.
mpirun_check:   fix: export MPICH_GPU_SUPPORT_ENABLED=1
mpirun_check:   or re-run with --recommended-settings to apply it
```

The 3-queues-per-rank figure above assumes GPU-aware MPI is actually on. With it off
the transport layer never initializes and a rank costs less, but it is also not doing
the thing you built it for.

It reads `NRANKS` from `-n`, `-np`, `--n`, `--np`, `--ntasks` and
`--ntasks-per-node`, falling back to `SLURM_NTASKS`; counts logical GPUs from
`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `SLURM_GPUS_ON_NODE` or the KFD topology;
takes the queue budget from the GPU family; and gets the partition mode from
`rocm-smi`, falling back to compute units per logical GPU on gfx942 only, since a small
compute-unit count means a slice there but simply a small GPU elsewhere. When it cannot
work out the rank count or the GPU count it says so and launches unchanged rather than
guessing. Note that mpirun's `-c` spelling of `-n` is
deliberately not recognized, because it collides with `bash -c` and similar in the
application part of the command line.

### Round-robin or block?

`--set-affinity` hands out devices round-robin, so rank *k* gets GPU *k* mod *N* and
neighbouring ranks land on different devices. The alternative is block, where
consecutive ranks share a device; `--affinity-order=block` selects it. The two differ
by more than enough to care about, and not in the direction intuition suggests.

Ghost Exchange `Ver6`, 8000x8000, 200 iterations, 4 MI300A GPUs, median of three runs.
`Total` in seconds, for each process grid `nprocx` x `nprocy`:

| ranks, grid | HIP round-robin | HIP block | OpenMP round-robin | OpenMP block |
|---|---|---|---|---|
| 12, 4x3 | 0.166 | **0.151** | **0.265** | 0.284 |
| 12, 3x4 | **0.110** | 0.156 | 0.257 | **0.226** |
| 12, 2x6 | **0.092** | 0.134 | 0.238 | 0.233 |
| 12, 6x2 | **0.103** | 0.158 | 0.285 | **0.272** |
| 8, 4x2 | **0.137** | 0.161 | 0.263 | **0.247** |
| 8, 2x4 | **0.091** | 0.132 | 0.226 | **0.193** |
| 16, 4x4 | 0.154 | 0.152 | 0.302 | **0.262** |

The two languages prefer opposite orders. For HIP round-robin wins five of seven, by 18
to 53%, and for OpenMP block wins five of seven, by 5 to 15%. Both patterns reproduce
cleanly: the three repeats of a cell never overlap the other order's three.

The obvious explanation does not survive contact with the data. Ghost Exchange numbers
ranks row-major, `xcoord = rank % nprocx`, so which neighbour pairs end up on the same
device is a function of the order and the grid, and it is easy to count them:

| ranks, grid | pairs co-located, round-robin | pairs co-located, block |
|---|---|---|
| 12, 4x3 | 8 of 17 | 6 of 17 |
| 12, 3x4 | 0 of 17 | 8 of 17 |
| 12, 2x6 | 0 of 16 | 8 of 16 |
| 8, 2x4 | 0 of 10 | 4 of 10 |
| 8, 4x2 | 4 of 10 | 4 of 10 |
| 16, 4x4 | 12 of 24 | 12 of 24 |

For HIP the correlation is backwards from the usual locality argument — the order that
co-locates *fewer* partners is the faster one — which is consistent with ranks sharing
an oversubscribed GPU having to take turns on it, so an exchange between two of them
serialises where an exchange across devices overlaps. But it is not the whole story:
the 8-rank 4x2 and 16-rank 4x4 rows co-locate the same number of pairs under both
orders, and 4x2 still differs by 18%. Something beyond the pair count — probably that
round-robin co-locates the contiguous top/bottom halos while block co-locates the
strided left/right ones — is also in play, and this example does not isolate it.

The practical upshot is that the order is worth a factor of 1.5 and is not predictable
from first principles, so it is an option rather than a fixed choice. Round-robin is
the default because it is the safer of the two under HIP, where the downside of
guessing wrong is largest. Try both on your own decomposition.

## OpenMP offload is not the same

A rank using OpenMP target offload holds `GPU_MAX_HW_QUEUES + 2` compute queues rather
than `+ 1`, because the OpenMP runtime keeps a stream of its own, so the rank ceilings
below the default differ. The companion example
[`../GhostExchange_Oversubscription_OpenMP`](../GhostExchange_Oversubscription_OpenMP)
measures that case. `mpirun_check.sh` is the same script in both and picks the right
model from the executable, so you do not have to choose.

## What to do

Count `GPU_MAX_HW_QUEUES + 1` queues per rank and keep the product with
`NRANKS` at or under 24, with a further cap of 6 ranks per CPX slice. Treat
`GPU_MAX_HW_QUEUES=4` as a number to lower rather than a safe default: 2 buys eight
ranks per SPX GPU and 1 buys twelve, and in this example neither costs anything,
because the Ghost Exchange versions use a single stream and cannot use a deeper pool
anyway. An application that genuinely overlaps work across streams will want the
opposite trade, and the way to settle it is to run `queue_probe` alongside your own code
rather than to reason from stream counts.

Assign devices explicitly, and then try both orders. `--set-affinity` is the cheap way
to be sure the ranks are where you think they are, and `--affinity-order=block` against
the round-robin default is worth up to a factor of 1.5 either way depending on your
decomposition.

Keep `HSA_XNACK` consistent across anything sharing a GPU. Mixing processes with XNACK
on and off is a third, undocumented trigger on this hardware:

```
Runlist is getting oversubscribed due to xnack on/off processes mixed on gfx9.
```

That one matters here because Ver6 allocates with `hipMalloc` and wants XNACK off, while
Ver1 and Ver8 rely on page migration and want it on, so co-scheduling them on one GPU
oversubscribes the runlist no matter how few queues each holds. `sweep.sh` sets it per
version for this reason.

Finally, read `dmesg` when a multi-rank GPU run disappoints. The driver names which
limit you crossed, which turns a vague slowdown into a one-line answer.

One caveat on reading it, learned the hard way here: the messages are not reliably
timestamped against the run that caused them. In runs isolated by 30 s of idle on both
sides, a message twice appeared about 36 s after a run that was comfortably inside both
budgets, and once named a trigger that configuration could not have hit. Treat `dmesg`
as telling you which limit this *node* is hitting under your workload, not as a
per-invocation verdict, and use run time against a smaller rank count as the real test.

## A note on ROCm versions

All numbers above are ROCm 7.2.2. The MI300A measurements were also taken under
6.4.3 and the two agree: the per-rank queue counts are identical, every under-budget
entry in the SPX sweep matches to within a few percent, and the CPX process boundary
sits at the same place (6 ranks 0.333 s and 7 ranks 0.803 s under 6.4.3, against
0.342 s and 0.671 s under 7.2.2). Nothing here appears to be a property of the ROCm
release, which also means the MI300A and MI250 results are directly comparable: both
were measured at 7.2.2.
