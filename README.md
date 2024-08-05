# AMD HPC Training Examples Repo

Welcome to AMD's HPC Training Examples Repo!
Here you will find a variety of examples to showcase the capabilities of AMD's GPU software stack.
Please be aware that the repo is continuously updated to keep up with the most recent releases of the AMD software.

## Repo Structure

Please refer to these instructions to locate the exercises you are interested in sorted by topic. 

1. `HIP`
   1. Basic Examples
      1. [`Stream_Overlap`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/Stream_Overlap): this example showcases how to share the workload of a GPU offload compation using several overlapping streams. The result is an additional gain in terms of time of execution due to the additional parallelism provided by the overlapping streams. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIP/Stream_Overlap/README.md).
      2. [`dgemm`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/dgemm): A simple (d)GEMM application created as an exercise to showcase simple matrix-matrix multiplications on AMD GPUs. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/dgemm/README.md).
      3. [`exercises`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/exercises): a collection of simple exercises such as device to host data transfer and basic GPU kernel implementation. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/exercises/README.md).
 

## Run the Tests

Most of the exercises in this repo can be run as a test suite by doing:

```
git clone https://github.com/amd/HPCTrainingExamples && \
cd HPCTrainingExamples && \
cd tests && \
./runTests.sh
```
You can also run a subset of the whole test suite by specifying the subset you are interested in as an input to the `runTests.sh` script. For instance: `./runTests.sh --pytorch`. To see a full list of the possible subsets that can be run, do `./runTests.sh --help`.

**NOTE**: tests can also be run manually from their respective directories, provided the necessary modules have been loaded and they have been compiled appropriately.

## Feedback
We welcome your feedback and contributions, feel free to use this repo to bring up any issues or submit pull requests.
The software made available here is released under the MIT license, more details can be found in `LICENSE.md`.
