/**
 * Driver for PCG with Incomplete Cholesky Preconditioner
 * ROCm 7.1.1 Implementation
 * 
 * Generates test matrix and invokes the PCG solver
 */

#include "pcg_ic.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error: " << hipGetErrorString(err)                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define ROCSPARSE_CHECK(call)                                                   \
    do {                                                                        \
        rocsparse_status status = call;                                         \
        if (status != rocsparse_status_success) {                               \
            std::cerr << "rocSPARSE error: " << status                          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define ROCBLAS_CHECK(call)                                                     \
    do {                                                                        \
        rocblas_status status = call;                                           \
        if (status != rocblas_status_success) {                                 \
            std::cerr << "rocBLAS error: " << status                            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/**
 * Create a 3D Laplacian matrix (7-point stencil) in CSR format
 * This is a standard SPD test matrix for iterative solvers
 */
void create_3d_laplacian(int nx, int ny, int nz,
                         std::vector<int>& row_ptr,
                         std::vector<int>& col_ind,
                         std::vector<double>& val,
                         std::vector<double>& b)
{
    int64_t n = (int64_t)nx * ny * nz;
    row_ptr.resize(n + 1);
    b.resize(n, 1.0);  // RHS = all ones

    // Count nnz per row
    row_ptr[0] = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int64_t row = i + j * nx + k * (int64_t)nx * ny;
                int count = 1;  // diagonal
                if (i > 0) count++;
                if (i < nx - 1) count++;
                if (j > 0) count++;
                if (j < ny - 1) count++;
                if (k > 0) count++;
                if (k < nz - 1) count++;
                row_ptr[row + 1] = row_ptr[row] + count;
            }
        }
    }

    int64_t nnz = row_ptr[n];
    col_ind.resize(nnz);
    val.resize(nnz);

    // Fill matrix entries (sorted column indices for each row)
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int64_t row = i + j * nx + k * (int64_t)nx * ny;
                int64_t idx = row_ptr[row];

                // Lower neighbors first (for IC ordering)
                if (k > 0) {
                    col_ind[idx] = row - (int64_t)nx * ny;
                    val[idx] = -1.0;
                    idx++;
                }
                if (j > 0) {
                    col_ind[idx] = row - nx;
                    val[idx] = -1.0;
                    idx++;
                }
                if (i > 0) {
                    col_ind[idx] = row - 1;
                    val[idx] = -1.0;
                    idx++;
                }

                // Diagonal
                col_ind[idx] = row;
                val[idx] = 6.0;
                idx++;

                // Upper neighbors
                if (i < nx - 1) {
                    col_ind[idx] = row + 1;
                    val[idx] = -1.0;
                    idx++;
                }
                if (j < ny - 1) {
                    col_ind[idx] = row + nx;
                    val[idx] = -1.0;
                    idx++;
                }
                if (k < nz - 1) {
                    col_ind[idx] = row + (int64_t)nx * ny;
                    val[idx] = -1.0;
                    idx++;
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    // Default problem size
    int nx = 20, ny = 20, nz = 20;
    double tol = 1e-8;
    int max_iter = 1000;

    // Parse command line arguments
    if (argc >= 4) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
    }
    if (argc >= 5) {
        tol = atof(argv[4]);
    }
    if (argc >= 6) {
        max_iter = atoi(argv[5]);
    }

    int64_t n = (int64_t)nx * ny * nz;
    std::cout << "========================================" << std::endl;
    std::cout << "PCG with IC(0) Preconditioner" << std::endl;
    std::cout << "ROCm 7.1.1 Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Problem size: " << nx << " x " << ny << " x " << nz 
              << " = " << n << " unknowns" << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "Max iterations: " << max_iter << std::endl;

    // Create test matrix (3D Laplacian)
    std::vector<int> h_row_ptr, h_col_ind;
    std::vector<double> h_val, h_b;
    create_3d_laplacian(nx, ny, nz, h_row_ptr, h_col_ind, h_val, h_b);

    int64_t nnz = h_row_ptr[n];
    std::cout << "Matrix nnz: " << nnz << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize ROCm device
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipSetDevice(0));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Using device: " << props.name << std::endl;
    std::cout << "========================================" << std::endl;

    // Create library handles
    rocsparse_handle sparse_handle;
    rocblas_handle blas_handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&sparse_handle));
    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle));

    // Allocate device memory
    int *d_row_ptr, *d_col_ind;
    double *d_val, *d_b, *d_x;

    HIP_CHECK(hipMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_ind, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_val, nnz * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x, n * sizeof(double)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_row_ptr,
                        h_row_ptr.data(),
                        (n + 1) * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_ind,
                        h_col_ind.data(),
                        nnz * sizeof(int),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_val,
                        h_val.data(),
                        nnz * sizeof(double),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b,
                        h_b.data(),
                        n * sizeof(double),
                        hipMemcpyHostToDevice));

    // Initialize x = 0 (initial guess)
    HIP_CHECK(hipMemset(d_x, 0, n * sizeof(double)));

    // Create timing events
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Solve the system
    int iter;
    double final_res;

    HIP_CHECK(hipEventRecord(start));

    pcg_incomplete_cholesky(sparse_handle,
                            blas_handle,
                            n,
                            nnz,
                            d_row_ptr,
                            d_col_ind,
                            d_val,
                            d_b,
                            d_x,
                            tol,
                            max_iter,
                            &iter,
                            &final_res);

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    // Print results
    std::cout << "========================================" << std::endl;
    std::cout << "Converged in " << iter << " iterations" << std::endl;
    std::cout << "Final residual: " << final_res << std::endl;
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    // Copy solution back to host (optional verification)
    std::vector<double> h_x(n);
    HIP_CHECK(hipMemcpy(h_x.data(),
                        d_x,
                        n * sizeof(double),
                        hipMemcpyDeviceToHost));

    // Cleanup device memory
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_ind));
    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_x));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Destroy library handles
    ROCSPARSE_CHECK(rocsparse_destroy_handle(sparse_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(blas_handle));

    return EXIT_SUCCESS;
}
