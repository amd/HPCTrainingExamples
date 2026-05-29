/*
 * Minimal MPI + OpenMP "hello" program used by the Affinity_MPI* tests.
 *
 * Output format matches the pass regexes in tests/CMakeLists.txt:
 *   "MPI %03d - OMP %03d\n"
 *
 * Each MPI rank spawns OMP_NUM_THREADS OpenMP threads, and every thread
 * prints one line with its (rank, thread_id) pair.
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int provided = 0;
    int rank = 0;
    int size = 0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("MPI %03d - OMP %03d\n", rank, tid);
    }

    MPI_Finalize();
    return 0;
}
