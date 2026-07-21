# Stream Overlap: Original

This is the original Stream Overlap example.

## Build and run

Ensure rocm/6.2 or later is loaded.

```bash
cd /path/to/Stream_Overlap/0-Orig
mkdir build
cd build
cmake ../
make -j

./compute_comm_overlap <num-of-streams>
```

## Profile using ROCm Systems Profiler

```bash
rocprof-sys-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
rocprof-sys-run -- ./compute_comm_overlap.inst <num-of-streams>
```
