// Counts the KFD user-mode queues owned by this rank at successive points in a
// GPU-aware MPI run, to measure how many hardware queues the MPI library adds.
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <dirent.h>
#include <unistd.h>
#include <cstdio>

// KFD queue types, from kfd_priv.h: only type 0 counts against the 24-queue
// compute-queue budget; SDMA queues are a separate resource.
enum { KFD_QUEUE_TYPE_COMPUTE = 0 };

static void count_queues(int *compute, int *other)
{
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/kfd/kfd/proc/%d/queues", getpid());
    *compute = 0;
    *other = 0;
    DIR *d = opendir(path);
    if (!d) {
        return;
    }
    for (struct dirent *e = readdir(d); e != nullptr; e = readdir(d)) {
        if (e->d_name[0] == '.') {
            continue;
        }
        char tpath[320];
        snprintf(tpath, sizeof(tpath), "%s/%s/type", path, e->d_name);
        FILE *f = fopen(tpath, "r");
        int type = -1;
        if (f) {
            if (fscanf(f, "%d", &type) != 1) {
                type = -1;
            }
            fclose(f);
        }
        if (type == KFD_QUEUE_TYPE_COMPUTE) {
            (*compute)++;
        } else {
            (*other)++;
        }
    }
    closedir(d);
}

// Before MPI_Init there is no communicator to synchronize on, so the early
// samples print unsynchronized and are tagged with the pid.
static void report_early(const char *label)
{
    int compute, other;
    count_queues(&compute, &other);
    printf("pid %d  compute_queues=%2d  other_queues=%2d  %s\n", getpid(), compute, other, label);
    fflush(stdout);
}

static void report(const char *label, int rank)
{
    int compute, other;
    count_queues(&compute, &other);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank %d  compute_queues=%2d  other_queues=%2d  %s\n", rank, compute, other, label);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void touch(double *p)
{
    p[threadIdx.x] = threadIdx.x;
}

int main(int argc, char **argv)
{
    // Controls: establish what the HIP runtime alone costs, before MPI exists.
    report_early("at start, before any HIP call");
    hipSetDevice(0);
    hipFree(nullptr);
    report_early("after HIP runtime init, before MPI_Init");

    double *pre;
    hipMalloc(&pre, 1024);
    hipLaunchKernelGGL(touch, dim3(1), dim3(64), 0, 0, pre);
    hipDeviceSynchronize();
    report_early("after kernel on default stream, before MPI_Init");

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    report("after MPI_Init", rank);

    double *dbuf, *rbuf;
    hipMalloc(&dbuf, 8u << 20);
    hipMalloc(&rbuf, 8u << 20);
    hipStream_t s;
    hipStreamCreate(&s);
    hipLaunchKernelGGL(touch, dim3(1), dim3(64), 0, s, dbuf);
    hipStreamSynchronize(s);

    report("after app kernel on 1 app stream", rank);

    int peer = (rank + 1) % nranks;
    MPI_Sendrecv(dbuf, 64, MPI_DOUBLE, peer, 0, rbuf, 64, MPI_DOUBLE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    report("after GPU-aware Sendrecv, 512 B (staged path)", rank);

    MPI_Sendrecv(dbuf, 1u << 20, MPI_DOUBLE, peer, 1, rbuf, 1u << 20, MPI_DOUBLE, peer, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    report("after GPU-aware Sendrecv, 8 MB (device path)", rank);

    MPI_Allreduce(MPI_IN_PLACE, dbuf, 1u << 20, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    report("after GPU-aware Allreduce, 8 MB", rank);

    MPI_Finalize();
    return 0;
}
