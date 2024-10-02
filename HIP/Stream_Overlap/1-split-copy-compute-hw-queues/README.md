This Stream Overlap example splits the H2D (and D2H) data copies and compute kernel
launches across streams. This is to enable the data copies to overlap across
multiple streams.

## Build and run
```
cd /path/to/Stream_Overlap/1-split-copy-hw-queues
mkdir build
cd build
cmake ../
make -j

1. Run baseline.
   ./compute_comm_overlap <num-of-streams>

2. Run with maximum HW queues per device.
   export GPU_MAX_HW_QUEUES=8 # for more than 4 streams
   ./compute_comm_overlap <num-of-streams>

3. Run with larger block size.
   ./compute_comm_overlap <num-of-streams> <block-size (optional, default:64)>
```

## Profile using Omnitrace (1.11.3)
```
omnitrace-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
omnitrace-run -- ./compute_comm_overlap.inst <num-of-streams>
```
