# Stream Overlap: Pageable memory

This Stream Overlap example uses pageable memory H2D data copy across multiple streams.

## Build and run

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

## Profile using Omnitrace (1.11.3)

```bash
omnitrace-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
omnitrace-run -- ./compute_comm_overlap.inst <num-of-streams>
```
