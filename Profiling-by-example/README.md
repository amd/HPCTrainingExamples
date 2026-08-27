# Profiling by example

In this directory, you will find guided examples that walk you through the functionalities
of the profilers available for AMD Instinct&trade; GPUs, and show you how the information
they report guides the optimizations you make.

Each example is a hands-on walkthrough rather than a tool reference: you can run every
command as it appears in the README. They share the same loop, which is to run a profiler,
read what it says, make one change, and measure again.

## [Jacobi](jacobi)

A tour of `rocprofv3`, `rocprof-sys`, and `rocprof-compute` applied to the MPI + HIP Jacobi
solver in [`HIP/jacobi`](../HIP/jacobi), with the commands written for Frontier. It answers a
sequence of questions about an application you are assumed to know nothing about: is it
CPU bound or GPU bound, what does the timeline look like across ranks, and what limits the
hottest kernel.

## [Shallow-water](shallow-water)

A tour of `rocprofv3`, `rocprof-sys`, `rocprof-compute` and its viewer, the Roofline
Extractor, and the ROCprof Trace Decoder, applied to a 2D shallow-water solver. It is
presented as a sequence of self-contained stages where every optimization is motivated by a
specific profiler observation. Each stage builds and runs on its own and prints its own
throughput and correctness checks, so you can reproduce the whole progression without
editing any source. It was also set up to showcase some of the newer capabilities of the
profiling tools: iteration multiplexing for counter collection, how a roofline guides
the next optimization, and how to obtain and visualize advanced thread traces.

- [`novice`](shallow-water/novice): a single GPU.
- [`advanced`](shallow-water/advanced): several GPUs with MPI.
