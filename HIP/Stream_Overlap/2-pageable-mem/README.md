# Stream Overlap: Pageable memory

This Stream Overlap example uses pageable memory H2D data copy across multiple streams.

## Build and run

Ensure rocm/6.2 or later is loaded.

```bash
cd /path/to/Stream_Overlap/2-pageable-mem
mkdir build
cd build
cmake ../
make -j

./compute_comm_overlap <num-of-streams>

# Unfied shared memory run
HSA_XNACK=1 ./compute_comm_overlap <num-of-streams>
```

## Profile using ROCm Systems Profiler

```bash
rocprof-sys-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
rocprof-sys-run -- ./compute_comm_overlap.inst <num-of-streams>
```
