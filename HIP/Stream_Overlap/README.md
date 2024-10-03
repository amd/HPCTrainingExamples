# Stream Overlap Example

This example is based on example 2 from Chapter 6 of the HIP Book: "Accelerated Computing with HIP", by Yifan Sun, Trinayan Baruah, and David R. Kaeli. The example demonstrates how to overlap data transfer and computation using HIP streams. The included directories step through different versions of the example. Each directory contains a `README.md` file that includes a description of the version and instructions for building and running the example.

This multi-streamed example is traced with [Omnitrace](https://rocm.docs.amd.com/projects/omnitrace/en/latest/doxygen/html/index.html#omnitrace). Omnitrace is now available in ROCm 6.2.0+ version package directly and does not need to be installed separately anymore. The figures included in the `figs` directory are generated using `Omnitrace v.1.11.3`. The command line trace instructions are included in the `README.md` file in each directory.

## Folder `0-Orig`

This is the original version of the example. It demonstrates the basic structure of the example and provides a starting point for the other versions. The memory copies and kernel execution are done together sequentially in each of the multiple streams.

## Folder `1-split-copy-compute-hw-queues`

This version of the example splits the host to device (and vice versa) memory copies and the kernel execution into separate loops over multiple streams. This allows for overlap of memory copies across across multiple streams in addition to overlap of kernel computations over multiple streams. This also enables overlap of data copies and kernel computations.

This example also exploits the environment variable controlling the GPU maximum hardware queues (`GPU_MAX_HW_QUEUES`) to achieve better performance for a multi-streamed application.

## Folder `2-pageable-mem`

This version of the example uses *pageable* memory for data transfers instead of pinned memory. This example is to demonstrate how pageable memory degrades performance of a multi-streamed application. Ideally, pinned memory should be used for data transfers in a multi-streamed application wherever possible (depending on available memory resources).

## Self-guided tour of the Stream Overlap example

The interested reader can follow these steps sequentially to understand the performance implications of use of multiple streams to overlap data transfers and kernel computations. The results shared in folder `figs` are obtained from running the example on an AMD Instinct MI250 single GCD.

1. Build the baseline example in `O-Orig` directory. The build and run instructions can be found in the `README.md` file in the directory.
2. Then run the example using multiple streams. Specifically choose 1, 2, and 4 streams and observe the performance improvements. Specifically note if the reduction in runtime scales linearly with the number of streams. See the figures in `figs/streams[1,2,4]_noQ_seq_copy.png` for reference.
3. Increase the number of streams to 8 and observe the performance degradation. This is because the GPU has a limited hardware resources and increasing the number of streams beyond the GPU's capability will degrade performance. See the figure in `figs/streams8_noQ_seq_copy.png` for reference.
4. Switch to `1-split-copy-compute-hw-queues` directory and build and run the example. Observe the performance improvements if any. Ideally, the performance improvement is only marginal. See the figure in `figs/streams8_noQ_split_copy.png` for reference.
5. Set the environment variable `GPU_MAX_HW_QUEUES` to 8 and observe the performance improvements. This is because the default number of hardware queues is 4. Increasing the number of hardware queues will improve the performance of a multi-streamed application, especially when the number of streams is more than the default number of hardware queues. Note that, the performance improvement is possible if the GPU resource is not yet fully saturated, for example, with limited register pressure, or limited shared memory usage. The performance improvement is clearly visible in the figure `figs/streams8_Q_split_copy.png`.
6. [Optional] Switch to `2-pageable-mem` directory and build and run the example. Observe the performance degradation due to use of pageable memory for data transfers. Ideally, pinned memory should be used for data transfers in a multi-streamed application
7. Repeat the above steps for a different GPU and observe the performance implications.
