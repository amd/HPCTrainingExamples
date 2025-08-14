#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

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

int main(int argc, char *argv[])
{
    std::cout << "AMD GPU ROCBlas Matrix Transpose Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;

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
        int height = size.first;
        int width = size.second;

        // Allocate host memory
        double* h_input = new double[height * width];
        double* h_output = new double[width * height];

        // Generate test data
        for (int i = 0; i < height * width; ++i) {
            h_input[i] = static_cast<double>(i % 1000);
        }

        // Allocate device memory
        double *d_input, *d_output;
        size_t input_size = height * width * sizeof(double);
        size_t output_size = width * height * sizeof(double);

        hipCheck( hipMalloc(&d_input, input_size) );
        hipCheck( hipMalloc(&d_output, output_size) );

        // Copy input data to device
        hipCheck( hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice) );

        std::cout << "\nTesting Matrix dimensions: " << height << " x " << width << std::endl;
        std::cout << "Input size: " << input_size / (1024.0 * 1024.0) << " MiB" << std::endl;
        std::cout << "Output size: " << output_size / (1024.0 * 1024.0) << " MiB" << std::endl;
        std::cout << "=========================================" << std::endl;

        // See https://github.com/ROCm/rocBLAS/blob/develop/clients/samples/example_c_dgeam.c
        //   for an example how to use the transpose library routine in rocblas

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
        std::cout << "ROCBlas Transpose - Average Time: " << time_rocblas << " μs" << std::endl;

        std::cout << "=========================================" << std::endl;

        // Copy result back to verify correctness (only for first version)
        hipCheck( hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost) );

        // Verify correctness
        bool is_correct = true;

        for (int i = 0; i < height && is_correct; ++i) {
            for (int j = 0; j < width; ++j) {
                if (h_input[i * width + j] != h_output[j * height + i]) {
                    is_correct = false;
                    break;
                }
            }
        }

        std::cout << "Verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;

        // Cleanup
        roc_status = rocblas_destroy_handle(handle);
        CHECK_ROCBLAS_STATUS(roc_status);

        hipCheck( hipFree(d_input) );
        hipCheck( hipFree(d_output) );

        delete[] h_input;
        delete[] h_output;
    }
}
