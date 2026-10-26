// Does GPU_MAX_HW_QUEUES govern OpenMP target offload the way it governs HIP?
#include <mpi.h>
#include <omp.h>
#include <dirent.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

enum { KFD_QUEUE_TYPE_COMPUTE = 0 };

static int count_compute()
{
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/kfd/kfd/proc/%d/queues", getpid());
    int n = 0;
    DIR *d = opendir(path);
    if (!d) return -1;
    for (struct dirent *e = readdir(d); e; e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        char t[320];
        snprintf(t, sizeof(t), "%s/%s/type", path, e->d_name);
        FILE *f = fopen(t, "r");
        int type = -1;
        if (f) { if (fscanf(f, "%d", &type) != 1) type = -1; fclose(f); }
        if (type == KFD_QUEUE_TYPE_COMPUTE) n++;
    }
    closedir(d);
    return n;
}

int main(int argc, char **argv)
{
    int nregions = (argc > 1) ? atoi(argv[1]) : 1;
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int N = 1 << 20;
    double *a = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) a[i] = i;

    #pragma omp target enter data map(to: a[0:N])
    int after_first = 0;
    for (int r = 0; r < nregions; r++) {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < N; i++) a[i] = a[i] * 1.000001 + r;
        if (r == 0) after_first = count_compute();
    }
    int after_app = count_compute();

    // GPU-aware point-to-point straight out of the mapped device buffer
    double *dev = nullptr;
    #pragma omp target data use_device_ptr(a)
    { dev = a; }
    int peer = (rank + 1) % nranks;
    MPI_Sendrecv(dev, 1 << 16, MPI_DOUBLE, peer, 0, dev, 1 << 16, MPI_DOUBLE, peer, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int after_p2p = count_compute();
    MPI_Allreduce(MPI_IN_PLACE, dev, 1 << 16, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int after_coll = count_compute();
    #pragma omp target exit data map(release: a[0:N])

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        const char *q = getenv("GPU_MAX_HW_QUEUES");
        const char *l = getenv("LIBOMPTARGET_AMDGPU_NUM_HSA_QUEUES");
        printf("  regions=%d  GPU_MAX_HW_QUEUES=%-5s LIBOMPTARGET_AMDGPU_NUM_HSA_QUEUES=%-5s "
               "first=%d app=%d +p2p=%d +coll=%d\n",
               nregions, q ? q : "unset", l ? l : "unset",
               after_first, after_app, after_p2p, after_coll);
        fflush(stdout);
    }
    MPI_Finalize();
    free(a);
    return 0;
}
