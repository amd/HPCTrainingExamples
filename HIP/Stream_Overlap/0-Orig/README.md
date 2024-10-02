# Stream Overlap: Original

This is the original Stream Overlap example.

## Build and run

```bash
cd /path/to/Stream_Overlap/0-Orig
mkdir build
cd build
cmake ../
make -j

./compute_comm_overlap <num-of-streams>
```

## Profile using Omnitrace (1.11.3)

```bash
omnitrace-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
omnitrace-run -- ./compute_comm_overlap.inst <num-of-streams>
```
