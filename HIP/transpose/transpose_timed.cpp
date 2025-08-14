#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#include <rocblas/rocblas.h>

#include "transpose_kernels.h"

// Macro for checking GPU API return values
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

#define CHECK_ROCBLAS_STATUS(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }

// Generic kernel launcher with timing
template<typename KernelFunc>
double benchmark_kernel(KernelFunc kernel, const double* __restrict d_input,
                      double* __restrict d_output,
                      int height, int width, const std::string& name,
                      int iterations = 5) {

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Warm up
    kernel<<<grid_size, block_size>>>(d_input, d_output, height, width);
    hipCheck( hipDeviceSynchronize() );

    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid_size, block_size>>>(d_input, d_output, height, width);
    }

    hipCheck( hipDeviceSynchronize() );
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float avg_time = duration.count() / static_cast<float>(iterations);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << name << " - Average Time: " << avg_time << " μs" << std::endl;

    return avg_time;
}

// Host function to launch all versions
void run_all_transpose_versions(double* h_input, double* h_output, int height, int width) {
    // Allocate device memory
    double *d_input, *d_output;
    size_t input_size = height * width * sizeof(double);
    size_t output_size = width * height * sizeof(double);

    hipCheck( hipMalloc(&d_input, input_size) );
    hipCheck( hipMalloc(&d_output, output_size) );

    // Copy input data to device
    hipCheck( hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice) );

    std::cout << "Matrix dimensions: " << height << " x " << width << std::endl;
    std::cout << "Input size: " << input_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Output size: " << output_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Benchmark all versions
    float time_basic_read_contiguous = benchmark_kernel(
        transpose_kernel_read_contiguous,
        d_input, d_output, height, width,
        "Basic Transpose, Read Contiguous"
    );

    float time_basic_write_contiguous = benchmark_kernel(
        transpose_kernel_write_contiguous,
        d_input, d_output, height, width,
        "Basic Transpose, Write Contiguous"
    );

    float time_tiled = benchmark_kernel(
        transpose_kernel_tiled,
        d_input, d_output, height, width,
        "Tiled Transpose"
    );

    // Create handle to rocblas library
    rocblas_handle handle;
    rocblas_status roc_status=rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(roc_status);

    // scalar arguments will be from host memory
    roc_status = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(roc_status);

    // set up the parameters needed for the transpose operation
    const double alpha = 1.0;
    const double beta  = 0.0;

    // For transpose: C= alpha * op(A) + beta * B
    // where op(A) = A^T and B is the zero matrix
    rocblas_operation transa = rocblas_operation_transpose;
    rocblas_operation transb = rocblas_operation_none;

    // Call rocblas_geam for the transpose operation
    roc_status =  rocblas_dgeam(handle,
                      transa, transb,
                      width, height,
                      &alpha, d_input, width,
                      &beta, d_output, width,
                      d_output, width);
    CHECK_ROCBLAS_STATUS(roc_status);

    hipCheck( hipDeviceSynchronize() );

    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    int iterations = 5;
    for (int i = 0; i < iterations; ++i) {
       roc_status =  rocblas_dgeam(handle,
                         transa, transb,
                         width, height,
                         &alpha, d_input, width,
                         &beta, d_output, width,
                         d_output, width);
       CHECK_ROCBLAS_STATUS(roc_status);
    }

    hipCheck( hipDeviceSynchronize() );
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float time_rocblas = duration.count() / static_cast<float>(iterations);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "rocBLAS Transpose - Average Time: " << time_rocblas << " μs" << std::endl;

    // Copy result back to verify correctness (only for first version)
    hipCheck( hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost) );

    // Cleanup
    hipCheck( hipFree(d_input) );
    hipCheck( hipFree(d_output) );

    std::cout << "=========================================" << std::endl;
    std::cout << "Performance Summary:" << std::endl;
    std::cout << "Basic read contiguous   " << time_basic_read_contiguous  << " μs" << std::endl;
    std::cout << "Basic write contiguous  " << time_basic_write_contiguous << " μs" << std::endl;
    std::cout << "Tiled - both contiguous " << time_tiled                  << " μs" << std::endl;
    std::cout << "rocBLAS                 " << time_rocblas                << " μs" << std::endl;

    std::cout << "\nApplication Bandwidth (Hardware bandwidth is 1.5x greater due to load/stores):" << std::endl;
    double TiB = (input_size+output_size)/1024.0/1024.0/1024.0/1024.0;
    std::cout << "Basic read contiguous   " << TiB / ((double)time_basic_read_contiguous/1.0e6)  << " TiB/sec" << std::endl;
    std::cout << "Basic write contiguous  " << TiB / ((double)time_basic_write_contiguous/1.0e6) << " TiB/sec" << std::endl;
    std::cout << "Tiled                   " << TiB / ((double)time_tiled/1.0e6)                  << " TiB/sec" << std::endl;
    std::cout << "rocBLAS                 " << TiB / ((double)time_rocblas/1.0e6)                << " TiB/sec" << std::endl;
    std::cout << std::endl;

    // Calculate speedup relative to basic version
    if (time_basic_write_contiguous > 0) {
        std::cout << "Speedup (Write Contiguous):        " << time_basic_read_contiguous / time_basic_write_contiguous << "x" << std::endl;
        std::cout << "Speedup (Tiled - Both Contiguous): " << time_basic_read_contiguous / time_tiled << "x" << std::endl;
        std::cout << "Speedup (rocBLAS):                 " << time_basic_read_contiguous / time_rocblas << "x" << std::endl;
    }
}

// Verification function to check correctness
bool verify_transpose(double* h_input, double* h_output, int height, int width) {
    bool correct = true;

    for (int i = 0; i < height && correct; ++i) {
        for (int j = 0; j < width; ++j) {
            if (h_input[i * width + j] != h_output[j * height + i]) {
                correct = false;
                break;
            }
        }
    }

    return correct;
}

// Generate test matrix
void generate_test_matrix(double* matrix, int height, int width) {
    for (int i = 0; i < height * width; ++i) {
        matrix[i] = static_cast<double>(i % 1000);
    }
}

// Main function with different test cases
int main() {
    std::cout << "AMD GPU Matrix Transpose Benchmark" << std::endl;
    std::cout << "===================================" << std::endl;

    // Test different matrix sizes
    std::vector<std::pair<int, int>> test_sizes = {
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2048, 2048},
        {4096, 4096},
        {8192, 8192}
    };

    for (const auto& size : test_sizes) {
        int height = size.first;
        int width = size.second;

        std::cout << "\nTesting " << height << " x " << width << " matrix:" << std::endl;

        // Allocate host memory
        double* h_input = new double[height * width];
        double* h_output = new double[width * height];

        // Generate test data
        generate_test_matrix(h_input, height, width);

        // Run all versions
        run_all_transpose_versions(h_input, h_output, height, width);

        // Verify correctness (only for the first test case)
        bool is_correct = verify_transpose(h_input, h_output, height, width);
        std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;

        // Cleanup
        delete[] h_input;
        delete[] h_output;
    }

    return 0;
}
