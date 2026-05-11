#!/bin/bash

# Minimal rocBLAS DGEMM smoke test.
#
# Builds and runs a small HIP/rocBLAS program (one DGEMM, 64x64x64,
# NN, double) against the currently loaded rocm module. Its purpose
# is to exercise the rocBLAS kernel-loading path -- which is where
# the "hipErrorInvalidImage/InvalidKernelFile" failures surface when
# Tensile kernels for the host GPU arch are missing from
# $ROCM_PATH/lib/rocblas/library/ -- without depending on optional
# ROCm packages.
#
# Why not rocblas-bench? It is not shipped by every distro-packaged
# ROCm release (e.g. on this cluster it is present in
# rocm-therock-23.{1,2}.0 and rocm-6.3.{0,1,2}, but absent in
# rocm-6.3.3+, all rocm-6.4.x, and all rocm-7.x). This script
# therefore avoids it and links to librocblas directly so the test
# runs on every rocm/* module.
#
# Why not hipblas? hipBLAS on AMD is a thin wrapper that ultimately
# dispatches to rocBLAS, so it would surface the same failure, but
# rocBLAS-direct keeps the dependency chain minimal and the failure
# attribution unambiguous.

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi
module list

if ! command -v hipcc >/dev/null 2>&1; then
   echo "ERROR: hipcc not on PATH after loading rocm" >&2
   exit 1
fi
if [ -z "${ROCM_PATH}" ]; then
   echo "ERROR: ROCM_PATH is not set; rocm module did not export it" >&2
   exit 1
fi

WORKDIR=$(mktemp -d -t rocblas_gemm_check_XXXXXXXXXX)
trap "rm -rf $WORKDIR" EXIT

cat > "$WORKDIR/rocblas_gemm_check.hip" << 'EOF'
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>

#define hipCheck(call)                                                    \
do {                                                                      \
   hipError_t gpuErr = call;                                              \
   if (gpuErr != hipSuccess) {                                            \
      std::fprintf(stderr, "HIP error - %s:%d: %s\n",                     \
                   __FILE__, __LINE__, hipGetErrorString(gpuErr));        \
      std::exit(1);                                                       \
   }                                                                      \
} while (0)

#define rocblasCheck(call)                                                \
do {                                                                      \
   rocblas_status s = (call);                                             \
   if (s != rocblas_status_success) {                                     \
      std::fprintf(stderr, "rocBLAS error - %s:%d: status=%d\n",          \
                   __FILE__, __LINE__, static_cast<int>(s));              \
      std::exit(30);                                                      \
   }                                                                      \
} while (0)

int main()
{
   constexpr int          N     = 64;
   constexpr std::size_t  bytes = static_cast<std::size_t>(N) * N * sizeof(double);
   double                *A     = nullptr;
   double                *B     = nullptr;
   double                *C     = nullptr;
   const  double          alpha = 1.0;
   const  double          beta  = 0.0;

   hipCheck(hipMalloc(reinterpret_cast<void**>(&A), bytes));
   hipCheck(hipMalloc(reinterpret_cast<void**>(&B), bytes));
   hipCheck(hipMalloc(reinterpret_cast<void**>(&C), bytes));
   hipCheck(hipMemset(A, 0, bytes));
   hipCheck(hipMemset(B, 0, bytes));
   hipCheck(hipMemset(C, 0, bytes));

   rocblas_handle h;
   rocblasCheck(rocblas_create_handle(&h));

   rocblasCheck(rocblas_dgemm(h,
                              rocblas_operation_none, rocblas_operation_none,
                              N, N, N,
                              &alpha, A, N, B, N,
                              &beta,  C, N));
   hipCheck(hipDeviceSynchronize());

   rocblas_destroy_handle(h);
   hipCheck(hipFree(A));
   hipCheck(hipFree(B));
   hipCheck(hipFree(C));

   std::printf("rocBLAS DGEMM check: SUCCESS (N=%d, NN, double)\n", N);
   return 0;
}
EOF

pushd "$WORKDIR" >/dev/null

echo "=== Compiling rocBLAS DGEMM smoke test ==="
hipcc -O2 -x hip rocblas_gemm_check.hip \
   -I${ROCM_PATH}/include \
   -L${ROCM_PATH}/lib -lrocblas \
   -o rocblas_gemm_check
rc=$?
if [ $rc -ne 0 ]; then
   echo "ERROR: hipcc build failed (rc=$rc)" >&2
   popd >/dev/null
   exit $rc
fi

echo "=== Running rocBLAS DGEMM smoke test ==="
./rocblas_gemm_check
rc=$?
popd >/dev/null
exit $rc
