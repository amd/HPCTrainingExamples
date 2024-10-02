# Stream Overlap: Split Copy and Compute across Streams

This Stream Overlap example splits the H2D (and D2H) data copies in multiple streams and compute kernel
launches into multiple streams into two separate loops. This is to enable the data copies to overlap across
multiple streams.

## Build and run

```bash
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

4. Serialize the runs across streams.
   export AMD_SERIALIZE_COPY=3
   export AMD_SERIALIZE_KERNEL=3
```

## Profile using Omnitrace (1.11.3)

```bash
omnitrace-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
omnitrace-run -- ./compute_comm_overlap.inst <num-of-streams>
```
