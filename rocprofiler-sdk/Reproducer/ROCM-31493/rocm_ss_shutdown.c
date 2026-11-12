/* ROCM-31493 reproducer.
 * HPCToolkit's sample-source framework (src/hpcrun/sample-sources/rocm-ss.c)
 * declares a method named `shutdown`. On ROCm 10 the rocprofiler-sdk public
 * headers transitively pull in <sys/socket.h>, which declares the POSIX
 * `int shutdown(int, int)`. The two collide:
 *   rocm-ss.c: error: conflicting types for 'shutdown'
 * This translation unit never calls the POSIX socket function; it only needs
 * its own `shutdown` identifier. The test checks that it compiles clean. */
#include <rocprofiler-sdk/rocprofiler.h>

/* local identifier that collides with the POSIX shutdown pulled in above */
static int shutdown(void) { return 0; }

int main(void) { return shutdown(); }
