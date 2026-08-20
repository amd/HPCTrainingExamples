
# Profiling by Example: 2D Shallow-Water Solver

README.md from `HPCTrainingExamples/Profiling-by-example/shallow-water` from the Training Examples repository.

Two guided, hands-on walkthroughs of profiling and optimizing the same HIP application on AMD
Instinct&trade; GPUs: one on a single GPU, one across several with MPI. Both start from a
straightforward implementation and improve it a step at a time, where every change is motivated by
something a profiling tool reported rather than by a rule of thumb. Every stage builds and runs on
its own and prints its own throughput and correctness checks, so you can reproduce the whole
progression without editing any source.

| Track | Scope | Tools it exercises |
|---|---|---|
| [`novice`](novice) | One GPU, five stages | `rocprofv3` kernel and HIP API traces, `OccupancyPercent` and `VALUBusy` counters, `rocprof-compute` and the Roofline Extractor |
| [`advanced`](advanced) | Several GPUs with MPI, seven stages | per-rank `rocprofv3`, `rocprof-sys` timelines, thread traces through the ROCprof Trace Decoder and Compute Viewer, rooflines |

Start with [`novice`](novice) unless you have already profiled single-process GPU code, which is what
[`advanced`](advanced) assumes. The novice track ends with a cumulative 5.50x speedup on one GPU; the
advanced track separates raw speed from scalability, and spends its second half on communication and
decomposition rather than on kernels.

## The application

The code solves the 2D shallow-water equations over a flat bed, which describe the depth and momentum
of a thin layer of fluid and are the standard model for phenomena like tsunami propagation and dam
breaks. Space is discretized with central finite differences, time is advanced with a classical
four-stage Runge-Kutta scheme, and everything is single precision. The initial condition is still
water with a Gaussian bump that spreads outward and reflects off the walls. Five kernels do the work,
of which `compute_rhs` is the hot one.

Both tracks use the same solver. The advanced one splits the global domain across MPI ranks, so each
rank owns a subdomain whose ghost ring has to be refreshed from its neighbours before every RK4
stage. Each track's README describes the application, its kernels and its build in full.

Performance is reported in MCUPS, millions of cell updates per second counting all four RK4 stages,
and correctness in two checks that every run prints: relative error in total mass, which should stay
near zero, and the minimum water depth, which must never go negative.

## Where the numbers come from

Every figure quoted in either track, timings and counters alike, was measured on an MI300A in SPX
mode. The novice numbers are medians of three runs; the advanced ones are single runs. Absolute
numbers will differ on other hardware, and part of the point of the exercise is that the best block
size and the size of each speedup are architecture-dependent. Treat every figure as something to
re-measure rather than to expect.
