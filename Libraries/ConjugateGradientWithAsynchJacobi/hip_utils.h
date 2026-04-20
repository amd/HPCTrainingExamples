#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#define HIP_CHECK(command)                                    \
{                                                             \
  hipError_t stat = (command);                                \
  if(stat != hipSuccess)                                      \
  {                                                           \
    std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
    " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
    exit(-1);                                                 \
  }                                                           \
}

#define ROCSPARSE_CHECK(command)                                               \
{                                                                              \
  rocsparse_status stat = (command);                                           \
  if(stat != rocsparse_status_success) {                                       \
    std::cerr << "rocSPARSE Reports Error " << rocsparse_get_status_name(stat) \
    << " at: " << __FILE__ << ":" << __LINE__                        \
    << std::endl;                                                    \
    exit(-1);                                                                  \
  }                                                                            \
}

#define ROCBLAS_CHECK(condition)                                                             \
    {                                                                                        \
        const rocblas_status status = condition;                                             \
        if(status != rocblas_status_success)                                                 \
        {                                                                                    \
            std::cerr << "rocBLAS error encountered: \"" << rocblas_status_to_string(status) \
                      << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;               \
            exit(-1);                                                      \
        }                                                                                    \
    }

