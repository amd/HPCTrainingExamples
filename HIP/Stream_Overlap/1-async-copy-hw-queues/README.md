This Stream Overlap example uses asynchronous H2D data copy across streams.

## Build and run
```
cd /path/to/Stream_Overlap/1-async-copy-hw-queues
mkdir build
cd build
cmake ../
make -j

export GPU_MAX_HW_QUEUES=8 # for more than 4 streams
./compute_comm_overlap <num-of-streams>
```

## Profile using Omnitrace (1.11.3)
```
omnitrace-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
omnitrace-run -- ./compute_comm_overlap.inst <num-of-streams>
```
