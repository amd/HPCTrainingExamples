
# Profiling by Example: 2D Shallow-Water Solver

README.md from `HPCTrainingExamples/Profiling-by-example/shallow-water` from the Training Examples repository.

A guided, hands-on walkthrough of profiling and optimizing a HIP application on a single AMD
Instinct&trade; GPU, in [`novice`](novice). It starts from a straightforward implementation and
improves it a step at a time, where every change is motivated by something a profiling tool reported
rather than by a rule of thumb. Six stages exercise `rocprofv3` kernel and HIP API traces, the
`OccupancyPercent` and `VALUBusy` counters, and rooflines from both `rocprof-compute` and the Roofline
Extractor, for a cumulative 5.50x speedup. Every stage builds and runs on its own and prints its own
throughput and correctness checks, so you can reproduce the whole progression without editing any
source.

## The application

The code solves the 2D shallow-water equations over a flat bed, which describe the depth and momentum
of a thin layer of fluid and are the standard model for phenomena like tsunami propagation and dam
breaks. Space is discretized with central finite differences, time is advanced with a classical
four-stage Runge-Kutta scheme, and everything is single precision. The initial condition is still
water with a Gaussian bump that spreads outward and reflects off the walls. Five kernels do the work,
of which `compute_rhs` is the hot one. The [`novice`](novice) README describes the application, its
kernels and its build in full.

Performance is reported in MCUPS, millions of cell updates per second counting all four RK4 stages,
and correctness in two checks that every run prints: relative error in total mass, which should stay
near zero, and the minimum water depth, which must never go negative.

## Where the numbers come from

Every figure quoted in the tutorial, timings and counters alike, was measured on an **MI300A** in SPX
mode, taking the median of three runs. Absolute numbers will differ on other hardware, and part of
the point of the exercise is that the best block size and the size of each speedup are
architecture-dependent. Treat every figure as something to re-measure rather than to expect.
