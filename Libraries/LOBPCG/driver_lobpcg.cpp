/**
 * Driver for LOBPCG Eigenvalue Solver
 * ROCm 7.1.1 Implementation
 * 
 * Reads matrix from Matrix Market file and computes smallest eigenvalues
 */

#include "lobpcg.h"
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>

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

#define ROCRAND_CHECK(call)                                                     \
    do {                                                                        \
        rocrand_status status = call;                                           \
        if (status != ROCRAND_STATUS_SUCCESS) {                                 \
            std::cerr << "rocRAND error: " << status                            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/**
 * Read Matrix Market file and convert to CSR format
 */
bool read_matrix_market(const std::string& filename,
                        int64_t& n,
                        std::vector<int>& row_ptr,
                        std::vector<int>& col_ind,
                        std::vector<double>& val)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    bool is_symmetric = false;
    bool is_pattern = false;

    // Read header
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            // Check for symmetric flag
            std::string lower_line = line;
            std::transform(lower_line.begin(),
                           lower_line.end(),
                           lower_line.begin(),
                           ::tolower);
            if (lower_line.find("symmetric") != std::string::npos) {
                is_symmetric = true;
            }
            if (lower_line.find("pattern") != std::string::npos) {
                is_pattern = true;
            }
            continue;
        }
        break;
    }

    // Parse dimensions
    int64_t m, nnz_file;
    std::istringstream iss(line);
    iss >> n >> m >> nnz_file;

    if (n != m) {
        std::cerr << "Error: Matrix must be square for eigenvalue problems" << std::endl;
        return false;
    }

    std::cout << "Matrix: " << n << " x " << m << ", nnz in file: " << nnz_file;
    if (is_symmetric) std::cout << " (symmetric)";
    if (is_pattern) std::cout << " (pattern)";
    std::cout << std::endl;

    // Read entries in COO format
    std::vector<int> coo_row, coo_col;
    std::vector<double> coo_val;
    coo_row.reserve(is_symmetric ? 2 * nnz_file : nnz_file);
    coo_col.reserve(is_symmetric ? 2 * nnz_file : nnz_file);
    coo_val.reserve(is_symmetric ? 2 * nnz_file : nnz_file);

    for (int64_t i = 0; i < nnz_file; i++) {
        int row, col;
        double value = 1.0;

        if (!(file >> row >> col)) {
            std::cerr << "Error reading entry " << i << std::endl;
            return false;
        }

        if (!is_pattern) {
            if (!(file >> value)) {
                std::cerr << "Error reading value for entry " << i << std::endl;
                return false;
            }
        }

        // Convert to 0-based indexing
        row--;
        col--;

        coo_row.push_back(row);
        coo_col.push_back(col);
        coo_val.push_back(value);

        // Add symmetric entry
        if (is_symmetric && row != col) {
            coo_row.push_back(col);
            coo_col.push_back(row);
            coo_val.push_back(value);
        }
    }

    int64_t nnz = coo_row.size();
    std::cout << "Total nnz (after symmetry expansion): " << nnz << std::endl;

    // Sort by row, then by column
    std::vector<int64_t> perm(nnz);
    for (int64_t i = 0; i < nnz; i++) perm[i] = i;
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (coo_row[a] != coo_row[b]) return coo_row[a] < coo_row[b];
        return coo_col[a] < coo_col[b];
    });

    // Convert to CSR
    row_ptr.resize(n + 1, 0);
    col_ind.resize(nnz);
    val.resize(nnz);

    for (int64_t i = 0; i < nnz; i++) {
        row_ptr[coo_row[perm[i]] + 1]++;
    }
    for (int64_t i = 0; i < n; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }

    for (int64_t i = 0; i < nnz; i++) {
        col_ind[i] = coo_col[perm[i]];
        val[i] = coo_val[perm[i]];
    }

    return true;
}

/**
 * Read initial vectors from Matrix Market file (optional)
 */
bool read_vectors_market(const std::string& filename,
                         std::vector<double>& vec,
                         int64_t& nrows,
                         int64_t& ncols)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;

    // Skip comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions
    std::istringstream iss(line);
    iss >> nrows >> ncols;

    vec.resize(nrows * ncols);

    // Read values (column-major order)
    for (int64_t j = 0; j < ncols; j++) {
        for (int64_t i = 0; i < nrows; i++) {
            if (!(file >> vec[j * nrows + i])) {
                std::cerr << "Error reading vector entry" << std::endl;
                return false;
            }
        }
    }

    return true;
}

/**
 * Generate random initial vectors directly on GPU using rocRAND
 * d_vec: device pointer to n*nev doubles
 */
void generate_random_vectors_gpu(double* d_vec,
                                 int64_t n,
                                 int nev,
                                 unsigned long long seed = 42)
{
    rocrand_generator gen;
    ROCRAND_CHECK(rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT));
    ROCRAND_CHECK(rocrand_set_seed(gen, seed));
    
    // Generate uniform doubles in [0, 1)
    ROCRAND_CHECK(rocrand_generate_uniform_double(gen, d_vec, n * nev));
    
    ROCRAND_CHECK(rocrand_destroy_generator(gen));
    
    std::cout << "Generated random initial vectors on GPU (seed=" << seed << ")" << std::endl;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <matrix.mtx> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -nev <value>      Number of eigenvalues to compute (default: 5)" << std::endl;
    std::cout << "  -tol <value>      Convergence tolerance (default: 1e-6)" << std::endl;
    std::cout << "  -maxiter <value>  Maximum iterations (default: 100)" << std::endl;
    std::cout << "  -x0 <file.mtx>    Initial vectors in Matrix Market format" << std::endl;
    std::cout << "                    (default: random vectors)" << std::endl;
    std::cout << "  -seed <value>     Random seed for initial vectors (default: 42)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << prog_name << " bcsstk14.mtx -nev 10 -tol 1e-8" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Parse arguments
    std::string matrix_file = argv[1];
    int nev = 5;
    double tol = 1e-6;
    int max_iter = 100;
    std::string x0_file = "";
    unsigned int seed = 42;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-nev") == 0 && i + 1 < argc) {
            nev = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "-maxiter") == 0 && i + 1 < argc) {
            max_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-x0") == 0 && i + 1 < argc) {
            x0_file = argv[++i];
        } else if (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "LOBPCG Eigenvalue Solver" << std::endl;
    std::cout << "ROCm 7.1.1 Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Matrix file: " << matrix_file << std::endl;
    std::cout << "Number of eigenvalues: " << nev << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "Max iterations: " << max_iter << std::endl;
    std::cout << "========================================" << std::endl;

    // Read matrix from file
    int64_t n;
    std::vector<int> h_row_ptr, h_col_ind;
    std::vector<double> h_val;

    if (!read_matrix_market(matrix_file, n, h_row_ptr, h_col_ind, h_val)) {
        return EXIT_FAILURE;
    }

    int64_t nnz = h_row_ptr[n];
    std::cout << "CSR nnz: " << nnz << std::endl;
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

    // Create handles
    rocsparse_handle sparse_handle;
    rocblas_handle blas_handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&sparse_handle));
    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle));

    // Allocate device memory
    int* d_row_ptr;
    int* d_col_ind;
    double* d_val;
    double* d_X;
    double* d_lambda;

    HIP_CHECK(hipMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_ind, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_val, nnz * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_X, n * nev * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_lambda, nev * sizeof(double)));

    // Copy matrix data to device
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

    // Create initial vectors
    bool use_random = true;
    if (!x0_file.empty()) {
        std::vector<double> h_X;
        int64_t nrows, ncols;
        if (read_vectors_market(x0_file, h_X, nrows, ncols) && nrows == n && ncols >= nev) {
            // Copy initial vectors from file to device
            HIP_CHECK(hipMemcpy(d_X,
                                h_X.data(),
                                n * nev * sizeof(double),
                                hipMemcpyHostToDevice));
            use_random = false;
            std::cout << "Loaded initial vectors from " << x0_file << std::endl;
        } else {
            std::cerr << "Failed to read initial vectors, using random" << std::endl;
        }
    }
    
    if (use_random) {
        // Generate random initial vectors directly on GPU using rocRAND
        generate_random_vectors_gpu(d_X, n, nev, seed);
    }

    std::cout << "========================================" << std::endl;

    // Run LOBPCG
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));

    int iter;
    lobpcg(sparse_handle,
           blas_handle,
           n,
           nnz,
           d_row_ptr,
           d_col_ind,
           d_val,
           nev,
           d_X,
           d_lambda,
           tol,
           max_iter,
           &iter,
           false);  // No preconditioner for now

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    // Copy results back
    std::vector<double> h_lambda(nev);
    HIP_CHECK(hipMemcpy(h_lambda.data(),
                        d_lambda,
                        nev * sizeof(double),
                        hipMemcpyDeviceToHost));

    std::cout << "========================================" << std::endl;
    std::cout << "LOBPCG completed in " << iter << " iterations" << std::endl;
    std::cout << "Time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Computed eigenvalues:" << std::endl;
    for (int i = 0; i < nev; i++) {
        std::cout << "  lambda[" << i << "] = " << h_lambda[i] << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_ind));
    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_lambda));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    ROCSPARSE_CHECK(rocsparse_destroy_handle(sparse_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(blas_handle));

    return EXIT_SUCCESS;
}
