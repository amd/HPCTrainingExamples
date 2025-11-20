#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <rocprofiler-sdk-roctx/roctx.h>

// Macro for checking GPU API return values
#define hipCheck(call)                                                                          \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

// Check XNACK/unified memory support using HIP device attributes
bool checkXNACKEnabled() {
    int deviceId;
    hipGetDevice(&deviceId);
    
    // Check if pageable memory access is supported (system allocator unified memory)
    int pageableMemoryAccess = 0;
    hipError_t status = hipDeviceGetAttribute(&pageableMemoryAccess, 
                                               hipDeviceAttributePageableMemoryAccess, 
                                               deviceId);
    
    if (status != hipSuccess) {
        std::cerr << "Warning: Could not query unified memory attributes, "
                  << "defaulting to explicit memory management" << std::endl;
        return false;
    }
    
    // XNACK/unified memory enabled if PageableMemoryAccess = 1
    // Note: ConcurrentManagedAccess is not required since we synchronize before CPU access
    bool xnackEnabled = (pageableMemoryAccess == 1);
    
    if (xnackEnabled) {
        std::cout << "Unified memory support detected (PageableMemoryAccess: " 
                  << pageableMemoryAccess << ")" << std::endl;
    } else {
        std::cout << "Unified memory not supported (PageableMemoryAccess: " 
                  << pageableMemoryAccess << "), using explicit memory management" << std::endl;
    }
    
    return xnackEnabled;
}

// init rocblas parameters
// dgemm performs C = alpha_gemm * op(A) * op(B) + beta_gemm * C
const static double alpha_dgemm = 1.0;
const static double beta_dgemm = 0.0;
// set the modes so that op(A)=A and op(B)=B (could be a transpose otherwise)
const static rocblas_operation op = rocblas_operation_none;

int main(int argc, char* argv[]) {
    // Check XNACK/unified memory support using HIP device attributes
    bool xnackEnabled = checkXNACKEnabled();
    
    if (xnackEnabled) {
        std::cout << "Using std::vector with unified memory" << std::endl;
    } else {
        std::cout << "Using explicit memory management (hipMalloc + hipMemcpy)" << std::endl;
    }

    // number of terms in truncated series
    int N = 20;
    // evaluating the solutions at t=0.5
    double t = 0.5;

    // Host memory for initialization
    std::vector<double> h_A_init(4);
    h_A_init[0] = -2.0;
    h_A_init[1] = -1.0;
    h_A_init[2] = 1.0;
    h_A_init[3] = -2.0;

    // Memory allocation based on XNACK status
    std::vector<double> h_A(4);  // For unified memory (XNACK enabled)
    double *d_A = nullptr;        // For explicit memory (XNACK disabled)
    double *h_A_explicit = nullptr; // Host copy for explicit memory

    if (xnackEnabled) {
        // Use std::vector directly (system allocator with unified memory)
        h_A[0] = h_A_init[0];
        h_A[1] = h_A_init[1];
        h_A[2] = h_A_init[2];
        h_A[3] = h_A_init[3];
    } else {
        // Use explicit memory management
        hipCheck( hipMalloc(&d_A, 4 * sizeof(double)) );
        h_A_explicit = new double[4];
        h_A_explicit[0] = h_A_init[0];
        h_A_explicit[1] = h_A_init[1];
        h_A_explicit[2] = h_A_init[2];
        h_A_explicit[3] = h_A_init[3];
        hipCheck( hipMemcpy(d_A, h_A_explicit, 4 * sizeof(double), hipMemcpyHostToDevice) );
    }

    // i=1 in the series
    double h_EXP[4] = {1.0 + h_A_init[0] * t, h_A_init[1] * t, 
                       h_A_init[2] * t, 1.0 + h_A_init[3] * t};

    // exact solution vector evaluated at t
    std::vector<double> x_exact(2);
    x_exact[0] = exp(-2.0*t)*cos(t);
    x_exact[1] = exp(-2.0*t)*sin(t);

#pragma omp parallel for reduction(+:h_EXP[:4]) schedule(dynamic)
    for(int i=2; i<N; i++){
        // init rocblas handle
        rocblas_handle handle;
        rocblas_create_handle(&handle);

        // Allocate per-thread memory based on XNACK status
        std::vector<double> h_powA(4);  // For unified memory (XNACK enabled)
        double *d_powA = nullptr;       // For explicit memory (XNACK disabled)
        double *h_powA_explicit = nullptr; // Host copy for explicit memory

        if (xnackEnabled) {
            // Use std::vector directly (system allocator with unified memory)
            h_powA[0] = h_A_init[0];
            h_powA[1] = h_A_init[1];
            h_powA[2] = h_A_init[2];
            h_powA[3] = h_A_init[3];
        } else {
            // Use explicit memory management
            hipCheck( hipMalloc(&d_powA, 4 * sizeof(double)) );
            h_powA_explicit = new double[4];
            h_powA_explicit[0] = h_A_init[0];
            h_powA_explicit[1] = h_A_init[1];
            h_powA_explicit[2] = h_A_init[2];
            h_powA_explicit[3] = h_A_init[3];
            hipCheck( hipMemcpy(d_powA, h_powA_explicit, 4 * sizeof(double), hipMemcpyHostToDevice) );
        }

        double denom = i;
        double num = t;
        
        for(int k=1; k<i; k++){
            roctxRangePush("rocblas_dgemm");
            
            // Use appropriate pointers based on XNACK status
            double *powA_ptr = xnackEnabled ? h_powA.data() : d_powA;
            double *A_ptr = xnackEnabled ? h_A.data() : d_A;
            
            rocblas_status status = rocblas_dgemm(handle, op, op, 2, 2, 2, 
                                                   &alpha_dgemm, powA_ptr, 2, A_ptr, 2, 
                                                   &beta_dgemm, powA_ptr, 2);
            roctxRangePop();
            hipCheck( hipDeviceSynchronize() );
            
            if (status != rocblas_status_success) {
                fprintf(stderr, "rocblas_dgemm failed with status %d\n", status);
            }
            denom *= k;
            num *= t;
        }
        
        // Synchronize before accessing results
        hipCheck( hipDeviceSynchronize() );
        
        // Copy results back to host (if using explicit memory)
        if (!xnackEnabled) {
            hipCheck( hipMemcpy(h_powA_explicit, d_powA, 4 * sizeof(double), hipMemcpyDeviceToHost) );
        }
        
        // reduction to array step
        for(int m=0; m<4; m++){
            double factor = num / denom;
            if (xnackEnabled) {
                // Unified memory: access vector directly after synchronization
                h_EXP[m] += h_powA[m] * factor;
            } else {
                // Explicit memory: use host copy
                h_EXP[m] += h_powA_explicit[m] * factor;
            }
        }
        
        if(i==2){
            int num_threads = omp_get_num_threads();
            std::cout << "Total num of threads is: " << num_threads << std::endl;
        }
        
        // Free memory
        if (!xnackEnabled) {
            hipCheck( hipFree(d_powA) );
            if (h_powA_explicit != nullptr) {
                delete[] h_powA_explicit;
            }
        }
        // std::vector automatically cleans up when going out of scope
        
        rocblas_destroy_handle(handle);
    }

    // Free shared memory
    if (!xnackEnabled) {
        hipCheck( hipFree(d_A) );
        if (h_A_explicit != nullptr) {
            delete[] h_A_explicit;
        }
    }
    // std::vector automatically cleans up when going out of scope

    // initial solution
    std::vector<double> x_0(2);
    x_0[0] = 1.0;
    x_0[1] = 0.0;

    // compute approx solution
    std::vector<double> x_approx(2);
    x_approx[0] = h_EXP[0]*x_0[0] + h_EXP[1]*x_0[1];
    x_approx[1] = h_EXP[2]*x_0[0] + h_EXP[3]*x_0[1];

    // compute L2 norm of error
    double norm = (x_exact[0] - x_approx[0])*(x_exact[0] - x_approx[0]);
    norm += (x_exact[1] - x_approx[1])*(x_exact[1] - x_approx[1]);
    norm = std::sqrt(norm);

    if(norm < 1.0e-12){
        std::cout << "PASSED!" << std::endl;
        std::cout << std::setprecision(16) << "L2 norm of error is: " << norm << std::endl;
    }
    else{
        std::cout << "FAILED!" << std::endl;
        std::cout << std::setprecision(16) << "L2 norm of error is larger than prescribed tolerance..." << norm << std::endl;
    }

    return 0;
}
