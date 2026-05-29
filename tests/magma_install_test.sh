#!/bin/bash

# This test verifies that MAGMA is installed correctly with HIP support
# by initializing the library, checking the backend, and running a GPU
# DGEMM if a device is available.

# NOTE: this test assumes MAGMA has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/magma_setup.sh

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load magma

WORKDIR=$(mktemp -d -t magma_test_XXXXXXXXXX)
trap "rm -rf $WORKDIR" EXIT

cat > "$WORKDIR/test_magma_hip.cpp" << 'EOF'
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "magma_v2.h"

int main(int argc, char** argv)
{
    magma_int_t status;

    printf("=== MAGMA HIP Installation Test ===\n\n");

    status = magma_init();
    if (status != MAGMA_SUCCESS) {
        printf("FAIL: magma_init() returned error %lld\n", (long long)status);
        return 1;
    }
    printf("PASS: magma_init() succeeded\n");

#if defined(MAGMA_HAVE_HIP)
    printf("PASS: Built with MAGMA_HAVE_HIP\n");
#elif defined(MAGMA_HAVE_CUDA)
    printf("FAIL: Built with CUDA backend, expected HIP\n");
    magma_finalize();
    return 1;
#else
    printf("FAIL: No GPU backend detected\n");
    magma_finalize();
    return 1;
#endif

    magma_int_t ndevices = 0;
    magma_device_t devices[MagmaMaxGPUs];
    magma_getdevices(devices, MagmaMaxGPUs, &ndevices);
    if (ndevices < 1) {
        printf("SKIP: No GPU devices found (run on a node with AMD GPUs to test compute)\n");
        printf("\n=== Installation verified (no GPU available for compute test) ===\n");
        magma_finalize();
        return 0;
    }
    printf("PASS: Found %lld GPU device(s)\n", (long long)ndevices);

    magma_setdevice(0);

    magma_int_t N = 256;
    magma_int_t lda = N;
    double alpha = 1.0, beta = 0.0;

    double *hA, *hB, *hC;
    magma_dmalloc_cpu(&hA, N * N);
    magma_dmalloc_cpu(&hB, N * N);
    magma_dmalloc_cpu(&hC, N * N);

    if (!hA || !hB || !hC) {
        printf("FAIL: Host memory allocation failed\n");
        magma_finalize();
        return 1;
    }

    for (magma_int_t j = 0; j < N; j++) {
        for (magma_int_t i = 0; i < N; i++) {
            hA[i + j * lda] = (i == j) ? 1.0 : 0.0;
            hB[i + j * lda] = (i == j) ? 1.0 : 0.0;
            hC[i + j * lda] = 0.0;
        }
    }

    double *dA, *dB, *dC;
    magma_dmalloc(&dA, N * N);
    magma_dmalloc(&dB, N * N);
    magma_dmalloc(&dC, N * N);

    if (!dA || !dB || !dC) {
        printf("FAIL: Device memory allocation failed\n");
        magma_free_cpu(hA); magma_free_cpu(hB); magma_free_cpu(hC);
        magma_finalize();
        return 1;
    }

    magma_queue_t queue;
    magma_queue_create(0, &queue);

    magma_dsetmatrix(N, N, hA, lda, dA, lda, queue);
    magma_dsetmatrix(N, N, hB, lda, dB, lda, queue);

    magma_dgemm(MagmaNoTrans, MagmaNoTrans, N, N, N,
                alpha, dA, lda, dB, lda, beta, dC, lda, queue);

    magma_dgetmatrix(N, N, dC, lda, hC, lda, queue);
    magma_queue_sync(queue);

    double max_err = 0.0;
    for (magma_int_t j = 0; j < N; j++) {
        for (magma_int_t i = 0; i < N; i++) {
            double expected = (i == j) ? 1.0 : 0.0;
            double err = fabs(hC[i + j * lda] - expected);
            if (err > max_err) max_err = err;
        }
    }

    if (max_err < 1e-12) {
        printf("PASS: GPU DGEMM (I * I = I) verified, max error = %e\n", max_err);
    } else {
        printf("FAIL: GPU DGEMM result incorrect, max error = %e\n", max_err);
        magma_queue_destroy(queue);
        magma_free(dA); magma_free(dB); magma_free(dC);
        magma_free_cpu(hA); magma_free_cpu(hB); magma_free_cpu(hC);
        magma_finalize();
        return 1;
    }

    magma_queue_destroy(queue);
    magma_free(dA); magma_free(dB); magma_free(dC);
    magma_free_cpu(hA); magma_free_cpu(hB); magma_free_cpu(hC);
    magma_finalize();

    printf("\n=== All tests PASSED ===\n");
    return 0;
}
EOF

pushd "$WORKDIR"

$ROCM_PATH/bin/hipcc -O2 -std=c++14 \
   -I$MAGMA_PATH/include -I$ROCM_PATH/include \
   -o test_magma_hip test_magma_hip.cpp \
   -L$MAGMA_PATH/lib -lmagma \
   -L$ROCM_PATH/lib -lhipblas -lhipsparse \
   -lopenblas \
   -Wl,-rpath,$MAGMA_PATH/lib -Wl,-rpath,$ROCM_PATH/lib

./test_magma_hip

popd
