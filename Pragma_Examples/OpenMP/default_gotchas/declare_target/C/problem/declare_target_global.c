/* PROBLEM
 *
 * `array` is a file-scope global that appears in `#pragma omp declare target`
 * (default `to`-style). It is therefore permanently in the device data
 * environment with infinite reference count, so map(to:..) and map(from:..)
 * without the `always` modifier silently skip every transfer.
 *
 * Expected (wrong) output:
 *     array[0] = 1.000000
 *     array[0] = 2.000000
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
