/* SOLUTION: -fopenmp-force-usm
 *
 * Source is identical to ../problem. The fix is in the Makefile, which adds
 * `-fopenmp-force-usm` to the compile line. That flag injects
 * `#pragma omp requires unified_shared_memory` into every translation unit.
 * Under USM the compiler emits an 8-byte device-side reference pointer in
 * place of a full device global, so `array` lives only in host memory and the
 * device accesses it directly through that pointer. There is nothing to
 * synchronize, so the map clauses are effectively no-ops.
 *
 * Requires `HSA_XNACK=1` at runtime (USM needs GPU page-fault servicing).
 *
 * Expected (correct) output:
 *     array[0] = 2.000000
 *     array[0] = 4.000000
 */

#include <stdio.h>

#pragma omp declare target
double array[100];
#pragma omp end declare target

int main(void) {
    for (int i = 0; i < 100; ++i) array[i] = 1.0;

    #pragma omp target enter data map(to:array)
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < 100; ++i) {
        array[i] = array[i] * 2.0;
    }
    #pragma omp target exit data map(from:array)

    printf("array[0] = %f\n", array[0]);

    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) {
        array[i] = array[i] * 2.0;
    }

    printf("array[0] = %f\n", array[0]);
    return 0;
}
