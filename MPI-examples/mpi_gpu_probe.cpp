// GPU-aware MPI reproducer for the ROCm 10.1 UCX/CMA "Bad address" failure.
//
// Exercises the same two code paths that abort in the CDash nightlies:
//   * point-to-point:  MPI_Sendrecv on device/managed buffers (Ghost Exchange
//                      191/194 -> ucp -> uct_cma_ep_tx -> process_vm_readv)
//   * collective:      MPI_Allreduce on a device/managed buffer via coll_ucc
//                      (UCC AllReduce 183 -> tl_ucp -> ucp -> uct_cma_ep_tx)
//
// argv[1] = allocation mode: 0 = hipMalloc (device),  1 = hipMallocManaged
// argv[2] = element count    (doubles; default 131072 => 1 MiB, forces rndv)
//
// Exit code 0 on success; nonzero (or SIGABRT from UCX) on failure.

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIPCHECK(x)                                                            \
   do {                                                                        \
      hipError_t _e = (x);                                                     \
      if (_e != hipSuccess) {                                                  \
         fprintf(stderr, "HIP error %s at line %d: %s\n", #x, __LINE__,        \
                 hipGetErrorString(_e));                                       \
         MPI_Abort(MPI_COMM_WORLD, 1);                                         \
      }                                                                        \
   } while (0)

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank = 0, size = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   int mode = (argc > 1) ? atoi(argv[1]) : 0;               // 0=device 1=managed
   const int N = (argc > 2) ? atoi(argv[2]) : 131072;        // doubles
   const size_t bytes = (size_t)N * sizeof(double);

   int ndev = 0;
   HIPCHECK(hipGetDeviceCount(&ndev));
   if (ndev < 1) {
      if (rank == 0) fprintf(stderr, "no GPUs visible\n");
      MPI_Abort(MPI_COMM_WORLD, 3);
   }
   HIPCHECK(hipSetDevice(rank % ndev));

   double *sbuf = nullptr, *rbuf = nullptr, *asum = nullptr;
   if (mode == 1) {
      HIPCHECK(hipMallocManaged(&sbuf, bytes));
      HIPCHECK(hipMallocManaged(&rbuf, bytes));
      HIPCHECK(hipMallocManaged(&asum, bytes));
      for (int i = 0; i < N; i++) sbuf[i] = (double)rank;
   } else {
      HIPCHECK(hipMalloc(&sbuf, bytes));
      HIPCHECK(hipMalloc(&rbuf, bytes));
      HIPCHECK(hipMalloc(&asum, bytes));
      double *tmp = (double *)malloc(bytes);
      for (int i = 0; i < N; i++) tmp[i] = (double)rank;
      HIPCHECK(hipMemcpy(sbuf, tmp, bytes, hipMemcpyHostToDevice));
      free(tmp);
   }
   HIPCHECK(hipDeviceSynchronize());
   MPI_Barrier(MPI_COMM_WORLD);

   // ---- point-to-point: ring sendrecv of device/managed buffers ----
   const int left = (rank - 1 + size) % size;
   const int right = (rank + 1) % size;
   MPI_Sendrecv(sbuf, N, MPI_DOUBLE, right, 0,
                rbuf, N, MPI_DOUBLE, left, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

   int p2p_ok = 1;
   {
      double *tmp = (double *)malloc(bytes);
      if (mode == 1) { HIPCHECK(hipDeviceSynchronize()); memcpy(tmp, rbuf, bytes); }
      else HIPCHECK(hipMemcpy(tmp, rbuf, bytes, hipMemcpyDeviceToHost));
      for (int i = 0; i < N; i++) if (tmp[i] != (double)left) { p2p_ok = 0; break; }
      free(tmp);
   }

   // ---- collective: allreduce (coll_ucc) of a device/managed buffer ----
   MPI_Allreduce(sbuf, asum, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   int coll_ok = 1;
   const double expect = (double)size * (size - 1) / 2.0;
   {
      double *tmp = (double *)malloc(bytes);
      if (mode == 1) { HIPCHECK(hipDeviceSynchronize()); memcpy(tmp, asum, bytes); }
      else HIPCHECK(hipMemcpy(tmp, asum, bytes, hipMemcpyDeviceToHost));
      for (int i = 0; i < N; i++) if (tmp[i] != expect) { coll_ok = 0; break; }
      free(tmp);
   }

   int all_p2p = 0, all_coll = 0;
   MPI_Reduce(&p2p_ok, &all_p2p, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&coll_ok, &all_coll, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

   int rc = (p2p_ok && coll_ok) ? 0 : 2;
   if (rank == 0) {
      printf("PROBE_RESULT mode=%s N=%d p2p=%s allreduce=%s\n",
             mode == 1 ? "managed" : "device", N,
             all_p2p ? "PASS" : "FAIL", all_coll ? "PASS" : "FAIL");
      fflush(stdout);
      if (!all_p2p || !all_coll) rc = 2;
   }
   MPI_Finalize();
   return rc;
}
