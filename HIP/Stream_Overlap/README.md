This example is based on example 2 from Chapter 6 of the HIP Book

Accelerated Computing with HIP
Yifan Sun, Trinayan Baruah, David R Kaeli

Available at Amazon and Barnes and Noble

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
