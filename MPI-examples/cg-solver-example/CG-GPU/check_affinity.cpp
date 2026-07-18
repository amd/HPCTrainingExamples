// =============================================================================
// Affinity / GPU-assignment diagnostic for the CG-GPU run.
//
// Reproduces the solver's device pick -- hipSetDevice(rank % device_count) --
// and reports, for every MPI rank:
//   node, global rank, node-local rank, the *_VISIBLE_DEVICES env, the visible
//   device count, the selected device ordinal, that device's PCI bus id (the
//   ground truth for which physical GPU it is), and the CPU affinity mask.
//
// If two ranks on the same node report the same PCI bus id, they are sharing a
// GPU.  Distinct bus ids => each rank has its own GPU.
//
// Build:  mpicxx -O2 -std=c++17 check_affinity.cpp -o check_affinity \
//                -I$ROCM_PATH/include -L$ROCM_PATH/lib -lamdhip64
// Run:    mpirun -n 4 ./check_affinity
// =============================================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <sched.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

static std::string cpu_affinity_string()
{
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return "unknown";

    // Compress the allowed CPU list into ranges: e.g. "0-23,48-71".
    std::string out;
    int n = CPU_SETSIZE, start = -1;
    char buf[64];
    for (int i = 0; i <= n; i++) {
        int in = (i < n) && CPU_ISSET(i, &set);
        if (in && start < 0) start = i;
        if (!in && start >= 0) {
            if (i - 1 == start) snprintf(buf, sizeof(buf), "%d", start);
            else                snprintf(buf, sizeof(buf), "%d-%d", start, i - 1);
            if (!out.empty()) out += ",";
            out += buf;
            start = -1;
        }
    }
    return out.empty() ? "none" : out;
}

static const char* env_or(const char* k, const char* dflt)
{
    const char* v = getenv(k);
    return (v && *v) ? v : dflt;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Node-local rank (same shared-memory domain == same node).
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);

    char host[256];
    gethostname(host, sizeof(host));

    int dev_count = 0;
    hipGetDeviceCount(&dev_count);

    // Mimic the solver's assignment exactly.
    int chosen = (dev_count > 0) ? (rank % dev_count) : -1;
    char busid[64] = "n/a";
    if (chosen >= 0) {
        hipSetDevice(chosen);
        int cur = -1;
        hipGetDevice(&cur);
        chosen = cur;
        hipDeviceGetPCIBusId(busid, sizeof(busid), cur);
    }

    std::string aff = cpu_affinity_string();

    char line[1024];
    snprintf(line, sizeof(line),
             "rank %2d  localrank %2d  node %-16s  ROCR=%-8s HIP=%-8s  "
             "dev_count=%d  chosen_dev=%d  pci=%s  cpus=[%s]",
             rank, local_rank, host,
             env_or("ROCR_VISIBLE_DEVICES", "unset"),
             env_or("HIP_VISIBLE_DEVICES", "unset"),
             dev_count, chosen, busid, aff.c_str());

    // Gather ordered lines on rank 0 for a clean, deterministic print.
    const int W = 1024;
    char* all = (rank == 0) ? (char*)malloc((size_t)W * size) : nullptr;
    MPI_Gather(line, W, MPI_CHAR, all, W, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("=== GPU assignment / affinity (mimics hipSetDevice(rank %% dev_count)) ===\n");
        for (int r = 0; r < size; r++)
            printf("%s\n", all + (size_t)r * W);

        // Flag same-node ranks that collide on the same PCI bus id.
        int collisions = 0;
        for (int a = 0; a < size; a++) {
            char na[256], pa[64]; long da;
            sscanf(all + (size_t)a * W, "rank %ld", &da);
            char* la = all + (size_t)a * W;
            char* nA = strstr(la, "node "); char* pA = strstr(la, "pci=");
            sscanf(nA, "node %255s", na); sscanf(pA, "pci=%63s", pa);
            for (int b = a + 1; b < size; b++) {
                char nb[256], pb[64];
                char* lb = all + (size_t)b * W;
                char* nB = strstr(lb, "node "); char* pB = strstr(lb, "pci=");
                sscanf(nB, "node %255s", nb); sscanf(pB, "pci=%63s", pb);
                if (!strcmp(na, nb) && !strcmp(pa, pb) && strcmp(pa, "n/a")) {
                    printf("WARNING: ranks %d and %d share GPU %s on %s\n",
                           a, b, pa, na);
                    collisions++;
                }
            }
        }
        if (collisions == 0)
            printf("OK: every rank has a distinct physical GPU.\n");
        free(all);
    }

    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
