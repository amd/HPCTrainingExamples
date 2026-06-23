// Minimal HIP example: three FP64 GEMM kernels (naive / mid / LDS-tiled) for rocprofv3 --att.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kN = 2048;
constexpr int kBlock = 16;
constexpr int kTile = 32;
constexpr int kSpotChecks = 16;

#define HIP_CHECK(call)                                                         \
    do {                                                                      \
        hipError_t err = (call);                                              \
        if (err != hipSuccess) {                                              \
            std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
                         hipGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

__global__ __launch_bounds__(kBlock * kBlock, 4)
void matmul_naive(const double* A, const double* B, double* C, int N) {
    const int row = blockIdx.y * kBlock + threadIdx.y;
    const int col = blockIdx.x * kBlock + threadIdx.x;
    if (row >= N || col >= N) {
        return;
    }
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

__global__ __launch_bounds__(kBlock * kBlock, 4)
void matmul_mid(const double* __restrict__ A,
                const double* __restrict__ B,
                double* __restrict__ C,
                int N) {
    const int row = blockIdx.y * kBlock + threadIdx.y;
    const int col = blockIdx.x * kBlock + threadIdx.x;
    if (row >= N || col >= N) {
        return;
    }
    double sum = 0.0;
#pragma unroll 4
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

__global__ __launch_bounds__(kTile * kTile, 2)
void matmul_fast(const double* A, const double* B, double* C, int N) {
    __shared__ double tile_a[kTile][kTile + 1];
    __shared__ double tile_b[kTile][kTile + 1];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;

    double sum = 0.0;
    for (int t = 0; t < (N + kTile - 1) / kTile; ++t) {
        const int k0 = t * kTile;
        const int a_col = k0 + threadIdx.x;
        const int b_row = k0 + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < N && a_col < N) ? A[row * N + a_col] : 0.0;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < N && col < N) ? B[b_row * N + col] : 0.0;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

double cpu_one_element(const std::vector<double>& A,
                       const std::vector<double>& B,
                       int N,
                       int row,
                       int col) {
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    return sum;
}

bool spot_check(const std::vector<double>& A,
                const std::vector<double>& B,
                const std::vector<double>& result,
                int N,
                double tol) {
    for (int s = 0; s < kSpotChecks; ++s) {
        const int idx = (s * 104729 + 17) % (N * N);
        const int row = idx / N;
        const int col = idx % N;
        const double expected = cpu_one_element(A, B, N, row, col);
        if (std::fabs(expected - result[idx]) > tol) {
            std::fprintf(stderr,
                         "mismatch at (%d,%d): expected %.17g got %.17g\n", row, col,
                         expected, result[idx]);
            return false;
        }
    }
    return true;
}

template <typename Kernel>
double bench_kernel(Kernel kernel,
                    const double* d_A,
                    const double* d_B,
                    double* d_C,
                    int N,
                    dim3 grid,
                    dim3 block,
                    int warmup,
                    int iters) {
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        HIP_CHECK(hipDeviceSynchronize());
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    const double sec =
        std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
    return (2.0 * static_cast<double>(N) * N * N) / (sec * 1.0e9);
}

int main() {
    const int N = kN;
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(double);

    std::vector<double> h_A(N * N);
    std::vector<double> h_B(N * N);
    std::vector<double> h_C(N * N);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>((i * 17 + 3) % 997) / 997.0;
        h_B[i] = static_cast<double>((i * 31 + 7) % 991) / 991.0;
    }

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::printf("Device: %s\n", props.name);
    std::printf("GEMM size: N=%d (FP64), block=%dx%d, tile=%d\n\n", N, kBlock,
                kBlock, kTile);

    const dim3 block(kBlock, kBlock);
    const dim3 grid((N + kBlock - 1) / kBlock, (N + kBlock - 1) / kBlock);
    const dim3 block_fast(kTile, kTile);
    const dim3 grid_fast((N + kTile - 1) / kTile, (N + kTile - 1) / kTile);

    const int warmup = 2;
    const int iters = 5;
    const double tol = 1.0e-9;

    const double g_naive =
        bench_kernel(matmul_naive, d_A, d_B, d_C, N, grid, block, warmup, iters);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));
    const bool ok_naive = spot_check(h_A, h_B, h_C, N, tol);
    std::printf("matmul_naive: %8.2f GFLOPS  %s\n", g_naive,
                ok_naive ? "PASS" : "FAIL");
    if (!ok_naive) {
        return EXIT_FAILURE;
    }

    const double g_mid =
        bench_kernel(matmul_mid, d_A, d_B, d_C, N, grid, block, warmup, iters);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));
    const bool ok_mid = spot_check(h_A, h_B, h_C, N, tol);
    std::printf("matmul_mid:   %8.2f GFLOPS  %s\n", g_mid, ok_mid ? "PASS" : "FAIL");
    if (!ok_mid) {
        return EXIT_FAILURE;
    }

    const double g_fast =
        bench_kernel(matmul_fast, d_A, d_B, d_C, N, grid_fast, block_fast, warmup, iters);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));
    const bool ok_fast = spot_check(h_A, h_B, h_C, N, tol);
    std::printf("matmul_fast: %8.2f GFLOPS  %s\n", g_fast, ok_fast ? "PASS" : "FAIL");
    if (!ok_fast) {
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    return EXIT_SUCCESS;
}
