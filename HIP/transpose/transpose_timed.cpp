#include "transpose_kernels.h"

// Generic kernel launcher with timing
template<typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, float* d_input, float* d_output,
                      int rows, int cols, const std::string& name,
                      int iterations = 5) {

    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    // Warm up
    kernel<<<grid_size, block_size>>>(d_input, d_output, rows, cols);
    hipCheck( hipDeviceSynchronize() );

    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid_size, block_size>>>(d_input, d_output, rows, cols);
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
void run_all_transpose_versions(float* h_input, float* h_output, int rows, int cols) {
    // Allocate device memory
    float *d_input, *d_output;
    size_t input_size = rows * cols * sizeof(float);
    size_t output_size = cols * rows * sizeof(float);

    hipCheck( hipMalloc(&d_input, input_size) );
    hipCheck( hipMalloc(&d_output, output_size) );

    // Copy input data to device
    hipCheck( hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice) );

    std::cout << "Matrix dimensions: " << rows << " x " << cols << std::endl;
    std::cout << "Input size: " << input_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Output size: " << output_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Benchmark all versions
    float time_basic_read_contiguous = benchmark_kernel(
        transpose_kernel_read_contiguous,
        d_input, d_output, rows, cols,
        "Basic Transpose, Read Contiguous"
    );

    float time_basic_write_contiguous = benchmark_kernel(
        transpose_kernel_write_contiguous,
        d_input, d_output, rows, cols,
        "Basic Transpose, Write Contiguous"
    );

    float time_basic = benchmark_kernel(
        transpose_lds_kernel,
        d_input, d_output, rows, cols,
        "Basic LDS Transpose"
    );

    float time_optimized = benchmark_kernel(
        transpose_lds_kernel_optimized,
        d_input, d_output, rows, cols,
        "Optimized LDS Transpose"
    );

    float time_coalesced = benchmark_kernel(
        transpose_lds_kernel_coalesced,
        d_input, d_output, rows, cols,
        "Coalesced LDS Transpose"
    );

    // Copy result back to verify correctness (only for first version)
    hipCheck( hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost) );

    // Cleanup
    hipCheck( hipFree(d_input) );
    hipCheck( hipFree(d_output) );

    std::cout << "=========================================" << std::endl;
    std::cout << "Performance Summary:" << std::endl;
    std::cout << "Basic readopt   " << time_basic_read_contiguous << " μs" << std::endl;
    std::cout << "Basic writeopt  " << time_basic_write_contiguous << " μs" << std::endl;
    std::cout << "Basic LDS:      " << time_basic << " μs" << std::endl;
    std::cout << "Optimized LDS:  " << time_optimized << " μs" << std::endl;
    std::cout << "Coalesced LDS:  " << time_coalesced << " μs" << std::endl;

    // Calculate speedup relative to basic version
    if (time_basic > 0) {
        std::cout << "Speedup (Write Opt): " << time_basic_read_contiguous / time_basic_write_contiguous << "x" << std::endl;
        std::cout << "Speedup (Basic LDS): " << time_basic_read_contiguous / time_basic << "x" << std::endl;
        std::cout << "Speedup (Optimized LDS): " << time_basic_read_contiguous / time_optimized << "x" << std::endl;
        std::cout << "Speedup (Coalesced LDS): " << time_basic_read_contiguous / time_coalesced << "x" << std::endl;
    }
}

// Verification function to check correctness
bool verify_transpose(float* h_input, float* h_output, int rows, int cols) {
    bool correct = true;

    for (int i = 0; i < rows && correct; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (h_input[i * cols + j] != h_output[j * rows + i]) {
                correct = false;
                break;
            }
        }
    }

    return correct;
}

// Generate test matrix
void generate_test_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(i % 1000);
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
        int rows = size.first;
        int cols = size.second;

        std::cout << "\nTesting " << rows << " x " << cols << " matrix:" << std::endl;

        // Allocate host memory
        float* h_input = new float[rows * cols];
        float* h_output = new float[cols * rows];

        // Generate test data
        generate_test_matrix(h_input, rows, cols);

        // Run all versions
        run_all_transpose_versions(h_input, h_output, rows, cols);

        // Verify correctness (only for the first test case)
        if (rows == 256 && cols == 256) {
            bool is_correct = verify_transpose(h_input, h_output, rows, cols);
            std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;
        }

        // Cleanup
        delete[] h_input;
        delete[] h_output;
    }

    return 0;
}
