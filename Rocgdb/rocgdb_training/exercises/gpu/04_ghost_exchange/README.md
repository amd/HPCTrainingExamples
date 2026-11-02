# GPU Exercise 4 (wrong results): cells that are never set

GhostExchange is a small halo-exchange ("ghost cell") MPI + HIP program from this repository. The version `Ver1WithBug` runs to the end without an error message, yet some of its values are wrong. Nothing crashes, so there is no fault to stop at: your job is to use `rocgdb` to find out **which** values are wrong and **why**.

The code is not copied here; it lives in this repository at `MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1WithBug` (from this directory: `../../../../../MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1WithBug`). Work on a copy, so the original stays buggy for the next person:

```sh
work_dir=$(mktemp ${HOME}/ghx_example_XXXXXXX)
mkdir -p ${work_dir}
cp -r ../../../../../MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1WithBug ${work_dir}
cd ${work_dir}
```


## Build for debugging

Build and run on a GPU node (see `../../README` for getting one and loading the toolchain):

```sh
module load rocm openmpi        # module names vary by site
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j
```

-   `-DCMAKE_BUILD_TYPE=Debug` gives the HIP code `-ggdb`. A plain `cmake ..` builds without debug information: `rocgdb` then finds no line numbers at all (try it later: `break GhostExchange.hip:35` says "No source file named GhostExchange.hip.").
-   The kernels write plain `malloc` memory, which needs demand paging (XNACK):

```sh
export HSA_XNACK=1
```


## See the wrong values

Run a grid small enough to read: 4x4 cells (`-i 4 -j 4`) on 4 ranks (`-x 4 -y 1`), one halo cell (`-h 1`), with the corners (`-c`), one iteration (`-I 1`), timed (`-t`), printing the grid (`-p`):

```sh
srun -n 4 --gpus=1 ./GhostExchange -x 4 -y 1 -i 4 -j 4 -h 1 -t -c -I 1 -p
```

-   Inside an allocation a plain `srun` is enough (it then uses the allocation's tasks); add `--mpi=pmix` if your site's default MPI type is not PMIx. Without Slurm, use `mpirun -np 4 ./GhostExchange ...`.
-   Look at the "Initial State" table. Each rank owns one column (`0`) between two halo columns (`-1`, `1`); rows `-1` and `4` are halo rows.
-   The kernel `init_core2` should set every owned cell to about 400 (`400.0 + 0.1 * (rank + row)`). Which rows have that, and which do not? What do the others hold instead?


## Narrow it down: values at breakpoints

It is not an MPI problem, so debug a single rank (`-x 1 -y 1`); an MPI program also starts without `srun`, as a one-rank job:

```sh
rocgdb --args ./GhostExchange -x 1 -y 1 -i 4 -j 4 -h 1 -t -c -I 1
```

-   Find, in `main`, where the two initialisation kernels (`init_core`, `init_core2`) are launched. Each launch is followed by a `hipDeviceSynchronize`; the first statement after it runs when that kernel is done:

```sh
grep -n 'init_core\|dim3' ../GhostExchange.hip
```

(`init_core` appears twice: with `-c` the first, conditional launch does not run.)

-   Break after each of the two, and print a cell of row 2 each time, and one of row 1 at the end:

```
(gdb) break GhostExchange.hip:<line after the first init>
(gdb) break GhostExchange.hip:<line after init_core2>
(gdb) run
(gdb) print x[2][0]
(gdb) continue
(gdb) print x[2][0]
(gdb) print x[1][0]
```

-   Did `init_core2` run at all? Did it write every row?


## Find the cause: how `init_core2` was launched

-   Read `init_core2` (near the top of `GhostExchange.hip`): how does each thread compute the row (`tidy`) and column (`tidx`) it writes?
-   Break *inside* the kernel, on the store (the line `x[tidy-jbegin][tidx-ibegin] = ...`); the breakpoint stays pending until `run` loads the GPU code:

```
(gdb) break GhostExchange.hip:<the store line>
(gdb) run
(gdb) info dispatches        # the launch: grid and workgroup sizes
(gdb) print threadIdx
(gdb) print blockDim
(gdb) print gridDim
```

-   With that launch, which values can `tidy` take? Which rows can the kernel write?
-   Compare with the launch in `main` (`dim3 grid2`, `dim3 block2`): the grid is sized for a different number of threads per workgroup in y than the block has.


## Fix and confirm

-   Make `block2` agree with the grid: 4 threads in y. Keep the original for `diff`:

```sh
cd ..
sed -i.orig 's/dim3 block2(64, 2, 1);/dim3 block2(64, 4, 1);/' GhostExchange.hip
diff -Naur GhostExchange.hip.orig GhostExchange.hip
make -C build -j
srun -n 4 --gpus=1 ./build/GhostExchange -x 4 -y 1 -i 4 -j 4 -h 1 -t -c -I 1 -p
```

-   Every owned cell now holds about 400 (rank 0: rows 2 and 3 hold `400.2` and `400.3`). Under `rocgdb`, the same breakpoint after `init_core2` now prints `400.19999999999999` (400.2) for `x[2][0]`.
-   The lesson: when results are wrong but nothing crashes, compare the launch you *meant* with the launch `rocgdb` shows you (`info dispatches`, `blockDim`, `gridDim`).
