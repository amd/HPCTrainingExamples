/**
 * Driver for PCG with Incomplete Cholesky Preconditioner
 * ROCm 7.1.1 Implementation
 * 
 * Reads matrix from Matrix Market file
 */

#include "pcg_ic.h"
#include <hip/hip_runtime.h>
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

// COO entry for sorting
struct COOEntry {
    int row;
    int col;
    double val;
    
    bool operator<(const COOEntry& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

/**
 * Read a Matrix Market file and convert to CSR format
 * Supports: real, symmetric/general, coordinate format
 * 
 * @param filename      Path to the .mtx file
 * @param n             Output: matrix dimension (assumes square)
 * @param row_ptr       Output: CSR row pointers
 * @param col_ind       Output: CSR column indices
 * @param val           Output: CSR values
 * @return              true on success, false on failure
 */
bool read_matrix_market(const std::string& filename,
                        int64_t& n,
                        std::vector<int>& row_ptr,
                        std::vector<int>& col_ind,
                        std::vector<double>& val)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    
    // Read header line
    if (!std::getline(file, line)) {
        std::cerr << "Error: Empty file" << std::endl;
        return false;
    }

    // Parse header
    bool is_symmetric = false;
    bool is_pattern = false;
    bool is_complex = false;
    
    // Convert to lowercase for comparison
    std::string header = line;
    std::transform(header.begin(), header.end(), header.begin(), ::tolower);
    
    if (header.find("%%matrixmarket") == std::string::npos) {
        std::cerr << "Error: Not a Matrix Market file" << std::endl;
        return false;
    }
    
    if (header.find("coordinate") == std::string::npos) {
        std::cerr << "Error: Only coordinate format is supported" << std::endl;
        return false;
    }
    
    if (header.find("complex") != std::string::npos) {
        std::cerr << "Error: Complex matrices are not supported" << std::endl;
        return false;
    }
    
    is_symmetric = (header.find("symmetric") != std::string::npos);
    is_pattern = (header.find("pattern") != std::string::npos);
    
    std::cout << "Matrix properties: ";
    std::cout << (is_symmetric ? "symmetric" : "general");
    std::cout << (is_pattern ? ", pattern" : ", real") << std::endl;

    // Skip comment lines
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    // Parse dimensions: rows cols nnz
    int64_t nrows, ncols, nnz_file;
    std::istringstream dims(line);
    dims >> nrows >> ncols >> nnz_file;
    
    if (nrows != ncols) {
        std::cerr << "Error: Matrix must be square for CG solver" << std::endl;
        return false;
    }
    
    n = nrows;
    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Entries in file: " << nnz_file << std::endl;

    // Read COO entries
    std::vector<COOEntry> coo_entries;
    coo_entries.reserve(is_symmetric ? 2 * nnz_file : nnz_file);

    int row, col;
    double value;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        if (is_pattern) {
            iss >> row >> col;
            value = 1.0;  // Pattern matrices get value 1.0
        } else {
            iss >> row >> col >> value;
        }
        
        // Matrix Market uses 1-based indexing
        row--;
        col--;
        
        COOEntry entry = {row, col, value};
        coo_entries.push_back(entry);
        
        // Add symmetric entry (if not on diagonal)
        if (is_symmetric && row != col) {
            COOEntry sym_entry = {col, row, value};
            coo_entries.push_back(sym_entry);
        }
    }
    
    file.close();

    int64_t nnz = coo_entries.size();
    std::cout << "Total entries (after symmetry expansion): " << nnz << std::endl;

    // Sort by (row, col)
    std::sort(coo_entries.begin(), coo_entries.end());

    // Convert COO to CSR
    row_ptr.resize(n + 1, 0);
    col_ind.resize(nnz);
    val.resize(nnz);

    // Count entries per row
    for (int64_t i = 0; i < nnz; i++) {
        row_ptr[coo_entries[i].row + 1]++;
    }

    // Prefix sum to get row pointers
    for (int64_t i = 0; i < n; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Fill column indices and values
    std::vector<int> row_count(n, 0);
    for (int64_t i = 0; i < nnz; i++) {
        int r = coo_entries[i].row;
        int64_t idx = row_ptr[r] + row_count[r];
        col_ind[idx] = coo_entries[i].col;
        val[idx] = coo_entries[i].val;
        row_count[r]++;
    }

    return true;
}

/**
 * Read a vector from Matrix Market file (array or coordinate format)
 * 
 * @param filename  Path to the .mtx file
 * @param vec       Output: vector values
 * @return          true on success, false on failure
 */
bool read_vector_market(const std::string& filename, std::vector<double>& vec)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open RHS file " << filename << std::endl;
        return false;
    }

    std::string line;
    
    // Read header line
    if (!std::getline(file, line)) {
        std::cerr << "Error: Empty RHS file" << std::endl;
        return false;
    }

    // Parse header
    std::string header = line;
    std::transform(header.begin(), header.end(), header.begin(), ::tolower);
    
    if (header.find("%%matrixmarket") == std::string::npos) {
        std::cerr << "Error: RHS file is not in Matrix Market format" << std::endl;
        return false;
    }

    bool is_array = (header.find("array") != std::string::npos);
    bool is_coordinate = (header.find("coordinate") != std::string::npos);
    bool is_pattern = (header.find("pattern") != std::string::npos);

    if (!is_array && !is_coordinate) {
        std::cerr << "Error: RHS must be in array or coordinate format" << std::endl;
        return false;
    }

    // Skip comment lines
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    // Parse dimensions
    std::istringstream dims(line);
    int64_t nrows, ncols;
    
    if (is_array) {
        // Array format: rows cols
        dims >> nrows >> ncols;
        if (ncols != 1) {
            std::cerr << "Warning: RHS has " << ncols << " columns, using first column only" << std::endl;
        }
        
        vec.resize(nrows);
        int64_t idx = 0;
        while (std::getline(file, line) && idx < nrows) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            iss >> vec[idx++];
        }
        
        std::cout << "Read " << idx << " RHS values from array format" << std::endl;
        
    } else {
        // Coordinate format: rows cols nnz
        int64_t nnz;
        dims >> nrows >> ncols >> nnz;
        
        vec.resize(nrows, 0.0);  // Initialize to zero
        
        int row, col;
        double value;
        int64_t count = 0;
        
        while (std::getline(file, line) && count < nnz) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            if (is_pattern) {
                iss >> row >> col;
                value = 1.0;
            } else {
                iss >> row >> col >> value;
            }
            // Matrix Market is 1-indexed
            if (row >= 1 && row <= nrows) {
                vec[row - 1] = value;
            }
            count++;
        }
        
        std::cout << "Read " << count << " RHS values from coordinate format" << std::endl;
    }

    file.close();
    return true;
}

/**
 * Generate RHS vector of all ones
 */
void generate_ones_rhs(std::vector<double>& vec, int64_t n)
{
    vec.resize(n);
    for (int64_t i = 0; i < n; i++) {
        vec[i] = 1.0;
    }
    std::cout << "Using RHS vector of all ones" << std::endl;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <matrix.mtx> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -tol <value>      Relative tolerance (default: 1e-8)" << std::endl;
    std::cout << "  -maxiter <value>  Maximum iterations (default: 1000)" << std::endl;
    std::cout << "  -rhs <file.mtx>   Right-hand side vector in Matrix Market format" << std::endl;
    std::cout << "                    (default: vector of all ones)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << prog_name << " bcsstk14.mtx -tol 1e-10 -maxiter 2000" << std::endl;
    std::cout << "  " << prog_name << " matrix.mtx -rhs rhs.mtx" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Parse arguments
    std::string matrix_file = argv[1];
    double tol = 1e-8;
    int max_iter = 1000;
    std::string rhs_file = "";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "-maxiter") == 0 && i + 1 < argc) {
            max_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-rhs") == 0 && i + 1 < argc) {
            rhs_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "PCG with IC(0) Preconditioner" << std::endl;
    std::cout << "ROCm 7.1.1 Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Matrix file: " << matrix_file << std::endl;
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

    // Create RHS vector
    std::vector<double> h_b;
    
    if (!rhs_file.empty()) {
        // Read RHS from Matrix Market file
        if (!read_vector_market(rhs_file, h_b)) {
            std::cerr << "Failed to read RHS file, using ones RHS" << std::endl;
            generate_ones_rhs(h_b, n);
        } else if ((int64_t)h_b.size() != n) {
            std::cerr << "Warning: RHS size (" << h_b.size() 
                      << ") differs from matrix size (" << n << ")" << std::endl;
            if ((int64_t)h_b.size() < n) {
                h_b.resize(n, 0.0);  // Pad with zeros
            }
        }
    } else {
        // Default: RHS of all ones
        generate_ones_rhs(h_b, n);
    }

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

    // Copy solution back to host
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
