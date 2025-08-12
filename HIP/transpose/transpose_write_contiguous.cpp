#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#include <hip/hip_runtime.h>

#define TILE_SIZE 32

__global__
void transpose_kernel_write_contiguous(double* __restrict input,
                                      double* __restrict output,
                                      const int rows,
                                      const int cols);

// Macro for checking GPU API return values
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

int main(int argc, char *argv[])
{
    std::cout << "AMD GPU Write Contiguous Matrix Transpose Benchmark" << std::endl;
    std::cout << "===================================================" << std::endl;

    int iterations = 5;

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

        // Allocate host memory
        double* h_input = new double[rows * cols];
        double* h_output = new double[cols * rows];

        // Generate test data
        for (int i = 0; i < rows * cols; ++i) {
            h_input[i] = static_cast<double>(i % 1000);
        }

        // Allocate device memory
        double *d_input, *d_output;
        size_t input_size = rows * cols * sizeof(double);
        size_t output_size = cols * rows * sizeof(double);

        hipCheck( hipMalloc(&d_input, input_size) );
        hipCheck( hipMalloc(&d_output, output_size) );

        // Copy input data to device
        hipCheck( hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice) );

        std::cout << "\nTesting Matrix dimensions: " << rows << " x " << cols << std::endl;
        std::cout << "Input size: " << input_size / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Output size: " << output_size / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "=========================================" << std::endl;

        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

        // Warm up
        transpose_kernel_write_contiguous<<<grid_size, block_size>>>(d_input, d_output, rows, cols);
        hipCheck( hipDeviceSynchronize() );

        // Time the kernel execution
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            transpose_kernel_write_contiguous<<<grid_size, block_size>>>(d_input, d_output, rows, cols);
        }

        hipCheck( hipDeviceSynchronize() );
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float time_basic_write_contiguous = duration.count() / static_cast<float>(iterations);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Basic Transpose, Write Contiguous - Average Time: " << time_basic_write_contiguous << " μs" << std::endl;

        std::cout << "=========================================" << std::endl;

        // Copy result back to verify correctness (only for first version)
        hipCheck( hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost) );

        // Verify correctness
        bool is_correct = true;

        for (int i = 0; i < rows && is_correct; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (h_input[i * cols + j] != h_output[j * rows + i]) {
                    is_correct = false;
                    break;
                }
            }
        }

        std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;

        // Cleanup
        hipCheck( hipFree(d_input) );
        hipCheck( hipFree(d_output) );

        delete[] h_input;
        delete[] h_output;
    }
}
