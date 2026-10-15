/* SOLUTION: declare target link(...)
 *
 * `link` keeps `array` out of the device data environment until it is
 * explicitly mapped, so the reference count starts at 0 and transitions
 * 0->1 on enter and 1->0 on exit. The OpenMP map rules then fire the
 * copies on those transitions; the `always` modifier is not needed.
 *
 * Works regardless of `HSA_XNACK`. With `HSA_XNACK=1` the runtime can
 * additionally use auto zero-copy to avoid the physical allocation/copy,
 * but correctness does not depend on it.
 *
 * Expected (correct) output:
 *     array[0] = 2.000000
 *     array[0] = 4.000000
 */

#include <stdio.h>

double array[100];
#pragma omp declare target link(array)

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
