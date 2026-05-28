/* SOLUTION: map(always, ...)
 *
 * `array` is still declare target with infinite reference count, but the
 * `always` map-type modifier overrides the OpenMP rule that skips the copy
 * when the variable is already in the device data environment. The runtime
 * now performs the host->device copy on enter and the device->host copy on
 * exit.
 *
 * Works regardless of `HSA_XNACK`.
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

    #pragma omp target enter data map(always, to: array)
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < 100; ++i) {
        array[i] = array[i] * 2.0;
    }
    #pragma omp target exit data map(always, from: array)

    printf("array[0] = %f\n", array[0]);

    #pragma omp parallel for
    for (int i = 0; i < 100; ++i) {
        array[i] = array[i] * 2.0;
    }

    printf("array[0] = %f\n", array[0]);
    return 0;
}
