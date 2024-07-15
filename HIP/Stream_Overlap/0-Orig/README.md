This is the original Stream Overlap example.

## Build and run
```
cd /path/to/Stream_Overlap
mkdir build
cd build
cmake ../
make -j

./compute_comm_overlap <num-of-streams>
```

## Profile using Omnitrace (1.11.3)
```
omnitrace-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
omnitrace-run -- ./compute_comm_overlap.inst <num-of-streams>
```
