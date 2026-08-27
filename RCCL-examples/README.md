
# RCCL Point-to-point and Collective

> [!NOTE]
> If you are on **AAC7**, replace the generic `module load openmpi rocm amdclang` steps with these modules to follow the rest of the exercise:
> ```
> module unload openmpi rocm
> module load rocm-therock/23.1.0 openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4-rocm-therock-23.1.0
> export CC=$(which amdclang); export CXX=$(which amdclang++); export FC=$(which amdflang)
> ```

## About these examples

[RCCL](https://github.com/ROCm/rccl) (ROCm Collective Communications Library) is AMD's GPU-optimized implementation of the NCCL API. It provides point-to-point (`ncclSend`/ `ncclRecv`) and collective (`ncclReduce`, `ncclAllReduce`, `ncclBroadcast`, ...) operations that move data directly between GPUs over Infinity Fabric / xGMI, bypassing the host and the MPI stack for the actual transfer.

`pt2pt.cpp` and `collective.cpp` in this directory are the RCCL counterparts of [`../MPI-examples/pt2pt.cpp`](../MPI-examples/pt2pt.cpp) and [`../MPI-examples/collective.cpp`](../MPI-examples/collective.cpp): same buffers, same initialization, same validation logic, but the actual data movement is done with RCCL calls instead of `MPI_Send`/`MPI_Recv`/`MPI_Reduce`. MPI is still used, but only to launch the ranks and to distribute the RCCL unique id needed to build the RCCL communicator (this is the same bootstrap idiom used by `rccl-tests` and by the RCCL backend of the OSU micro-benchmarks) — it never touches the GPU payload.

| Example | MPI equivalent | What it shows |
|---|---|---|
| `pt2pt.cpp` | `pt2pt.cpp` | Rank 0 sends a GPU buffer to rank 1 with `ncclSend`/`ncclRecv` |
| `collective.cpp` | `collective.cpp` | Every rank contributes a GPU buffer to a sum-reduction with `ncclReduce` |

Allocate at least two GPUs and set up your environment
```
module load rocm openmpi  # For modules on AAC7, see note at beginning
```

Find the code and compile
```
cd HPCTrainingExamples/RCCL-examples
make
```

Set the environment variable and run the code
```
mpirun -n 2 -mca pml ucx ./pt2pt
mpirun -n 2 -mca pml ucx ./collective
```

or simply

```
make test
```

## How the RCCL communicator is created

Both examples follow the same bootstrap pattern:

1. `MPI_Init`/`MPI_Comm_rank`/`MPI_Comm_size` give each process its rank and the total number of ranks, exactly like the MPI examples.
2. `hipSetDevice` binds each rank to one GPU.
3. Rank 0 calls `ncclGetUniqueId` to create an id for the new communicator, and `MPI_Bcast` distributes it to every rank.
4. Every rank calls `ncclCommInitRank(&comm, size, id, rank)` to join the same RCCL communicator.
5. A `hipStream_t` is created; all RCCL calls are asynchronous on this stream, so `hipStreamSynchronize` is used before touching the results on the host.

```c
ncclUniqueId id;
ncclComm_t comm;
if (rank == 0) ncclGetUniqueId(&id);
MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
ncclCommInitRank(&comm, size, id, rank);
```

## pt2pt.cpp

Rank 0 fills a 10-element `int` GPU buffer with `0..9` and sends it to rank 1 with `ncclSend`; rank 1 receives it with `ncclRecv` and checks that the values match:

```c
if (rank == 0) ncclSend(d_buf, bufsize, ncclInt, /*peer=*/1, comm, stream);
if (rank == 1) ncclRecv(d_buf, bufsize, ncclInt, /*peer=*/0, comm, stream);
```

Unlike `MPI_Send`/`MPI_Recv`, `ncclSend`/`ncclRecv` are enqueued on a stream and complete asynchronously; `hipStreamSynchronize(stream)` is called before rank 1 copies the buffer back to the host to validate it.

## collective.cpp

Every rank initializes a 10-element `int` GPU buffer with `0..9` and all ranks participate in a sum-reduction onto rank 0 (the root) with `ncclReduce`:

```c
ncclReduce(d_sendbuf, d_recvbuf, count, ncclInt, ncclSum, root, comm, stream);
```

Rank 0 then checks that `d_recvbuf[i] == i * size`, i.e. the sum of `i` contributed by every rank — the same check as `../MPI-examples/collective.cpp`.

## RCCL Test

See [`../MPI-examples/README.md`](../MPI-examples/README.md#rccl-test) for instructions on building and running `rccl-tests`, which benchmark the bandwidth and latency of these same RCCL operations (`sendrecv_perf`, `all_reduce_perf`, ...) instead of demonstrating their correctness.
