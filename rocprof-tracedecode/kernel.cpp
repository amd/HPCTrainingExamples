#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <chrono>
#include <iostream>

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

template<typename Type>
class Matrix
{
public:
    Matrix(int _rows, int _columns): rows(_rows), columns(_columns), memsize(_rows*_columns*sizeof(Type))
    {
        host = new Type[rows*columns];
        memset(host, 0, memsize);
        HIP_API_CALL(hipMalloc((void**)&dev, memsize));
    }

    ~Matrix()
    {
        HIP_API_CALL(hipDeviceSynchronize());
        hipFree(dev);
        delete[] host;
    }

    void toDevice()
    {
        HIP_API_CALL(hipMemcpy(dev, host, memsize, hipMemcpyDefault));
        HIP_API_CALL(hipDeviceSynchronize());
    }

    void toHost()
    {
        HIP_API_CALL(hipMemcpy(host, dev, memsize, hipMemcpyDefault));
        HIP_API_CALL(hipDeviceSynchronize());
    }

    const int rows;
    const int columns;
    const int memsize;

    Type* host;
    Type* dev;
};

#define SHMBLOCK 64
#define TBLOCK 16

using float16 = __hip_bfloat16;
using float8  = __hip_fp8_e4m3_fnuz;
using Vec4    = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using IVec4    = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using Vec16   = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using Vec8     = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using float8x8 = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;

static_assert(sizeof(float8) == sizeof(uint8_t));

inline __device__ uint64_t castfrom8x8(const float8x8& f)
{
    return *reinterpret_cast<const uint64_t*>(&f);
}

inline __device__ Vec4 mfma(const float8x8& a, const float8x8& b)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(castfrom8x8(a), castfrom8x8(b), Vec4{}, 0, 0, 0); 
}

inline __device__ Vec4 mfma(const float8x8& a0, const float8x8& a1, const float8x8& b0, const float8x8& b1)
{
    Vec4 ret = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(castfrom8x8(a0), castfrom8x8(b0), Vec4{}, 0, 0, 0);
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(castfrom8x8(a1), castfrom8x8(b1), ret, 0, 0, 0);
}

// HEIGHT <= 4
// WIDTH > 4 causes register spill
template<int WIDTH, int HEIGHT = 4>
__global__ void __launch_bounds__(TBLOCK*TBLOCK*2, 4)
fp8_gemm_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    float16* __restrict__ c,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int MDIM, int NDIM, int KDIM
) {
    const int X = blockIdx.x * SHMBLOCK * WIDTH;
    const int Y = blockIdx.y * TBLOCK * HEIGHT;
    
    const int TX = threadIdx.x%TBLOCK;
    const int TY = threadIdx.y%4;
    const int TZ = (threadIdx.y/4)%4;
    const int TYZ = threadIdx.y%16;
    const int TW = (threadIdx.y/16)%2;

    __shared__ uint8_t a_shared[WIDTH][4][4][4][TBLOCK][4];
    __shared__ uint8_t b_shared[4][HEIGHT][4][TBLOCK][4];
    __shared__ float   scalar[4][TBLOCK][WIDTH];

    if (TW != 0)
    {
        for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            int k0 = k1 + k2;
            IVec4 temp_a[WIDTH], temp_b;

            int tx3 = (TX & 3) << 2;
            int i1 = TX >> 2;

            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && Y + 4*TX < NDIM)
                temp_b[j1] = *reinterpret_cast<const uint32_t*>(&b[(k0 + TZ*TBLOCK + TY + 4*j1)*NDIM + Y + 4*TX]);

            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && X < MDIM)
            for (int n=0; n<WIDTH; n++)
                temp_a[n][j1] = (X + n*SHMBLOCK < MDIM) ? *reinterpret_cast<const uint32_t*>(&a[(k0 + j1*TBLOCK + TYZ)*MDIM + X + 4*TX + n*SHMBLOCK]) : 0;

            if (k0 != 0) __syncthreads();

            if (i1 < HEIGHT)
            for (int j1=0; j1<4; j1++)
            for (int m=0; m<4; m++)
                b_shared[TZ][i1][TY][tx3 + m][j1] = (temp_b[j1] >> (8*m)) & 0xFF;

            for (int j1=0; j1<4; j1++)
            for (int m=0; m<4; m++)
            for (int n=0; n<WIDTH; n++)
                a_shared[n][i1][TZ][TY][tx3 + m][j1] = (temp_a[n][j1] >> (8*m)) & 0xFF;

            __syncthreads();
        }
        return;
    }

    Vec16 reg_res[WIDTH];
    for (int n=0; n<WIDTH; n++) reg_res[n] = {};

    float tmp_a[WIDTH];
    float tmp_b;
    if (TZ == 0)
    {
        for (int n=0; n<WIDTH; n++)
            tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
        tmp_b = scale_b[Y/128];
    }

    for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
    {
        if (TZ == 0)
        {
            for (int n=0; n<WIDTH; n++) scalar[TY][TX][n] = tmp_b * tmp_a[n];

            if (k1 + 2*SHMBLOCK < KDIM)
            {
                for (int n=0; n<WIDTH; n++)
                    tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[(k1/128 + 1)*MDIM + X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
                tmp_b = scale_b[(k1/128 + 1)*((NDIM+127)/128) + (Y/128)];
            }
        }
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            int k0 = k1 + k2;

            __syncthreads();
            
            float8x8 a0_load[WIDTH], a1_load[WIDTH];
            float8x8 b0_load[HEIGHT], b1_load[HEIGHT];

            for (int m=0; m<8; m++)
            for (int n=0; n<HEIGHT; n++)
            {
                b0_load[n][m] = b_shared[m/4 + 0][n][TY][TX][m%4];
                b1_load[n][m] = b_shared[m/4 + 2][n][TY][TX][m%4];
            }

            for (int m=0; m<8; m++)
            for (int r=0; r<WIDTH; r++)
            {
                a0_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 0];
                a1_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 2];
            }

            Vec4 scal[WIDTH];
            for (int n=0; n<WIDTH; n++)
            for (int m=0; m<4; m++)
                scal[n][m] = scalar[TZ][TY*4 + m][n];

            __syncthreads();

            for (int n=0; n<HEIGHT; n++)
            for (int r1=0; r1<WIDTH; r1++)
            {
                Vec4 res = mfma(a0_load[r1], a1_load[r1], b0_load[n], b1_load[n]);

                for (int m=0; m<4; m++)
                    reg_res[r1][m*HEIGHT + n] += scal[r1][m] * res[m];
            }
        }
    }

    for (int j=0; j<4; j ++) for (int i=0; i<HEIGHT; i ++) for (int r1=0; r1<WIDTH; r1++)
    if (Y + i*TBLOCK < NDIM && 4*TYZ + X + r1*SHMBLOCK < MDIM)
        c[(4*TYZ + j + X + r1*SHMBLOCK)*NDIM + Y + i*TBLOCK + TX] = (float16) reg_res[r1][j*HEIGHT + i];
}

void launchHip(
    const void* a,
    const void* b,
    void* c,
    const float* scale_a,
    const float* scale_b,
    int M, int N, int K
) {
    dim3 block(TBLOCK, 32, 1);

    int mi300_threads = SHMBLOCK * SHMBLOCK * 304;
    int bpcu = M * N / mi300_threads;

    if (bpcu >= 8 && M >= 4*SHMBLOCK)
    {
        dim3 grid((M + 4*SHMBLOCK - 1) / SHMBLOCK / 4, N / SHMBLOCK);

        fp8_gemm_kernel<4><<<grid, block, 0, 0>>>(
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<float16*>(c),
            scale_a,
            scale_b,
            M, N, K);
    }
    else if (bpcu >= 3 && M >= 2*SHMBLOCK)
    {
        dim3 grid((M + 2*SHMBLOCK - 1) / SHMBLOCK / 2, N / SHMBLOCK);

        fp8_gemm_kernel<2><<<grid, block, 0, 0>>>(
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<float16*>(c),
            scale_a,
            scale_b,
            M, N, K);
    }
    else if (bpcu >= 1 || N > M || 2 * M * N > mi300_threads)
    {
        dim3 grid((M + SHMBLOCK - 1) / SHMBLOCK, N / SHMBLOCK);

        fp8_gemm_kernel<1><<<grid, block, 0, 0>>>(
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<float16*>(c),
            scale_a,
            scale_b,
            M, N, K);
    }
    else
    {
        dim3 grid((M + SHMBLOCK - 1) / SHMBLOCK, 2 * N / SHMBLOCK);

        fp8_gemm_kernel<1, 2><<<grid, block, 0, 0>>>(
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<float16*>(c),
            scale_a,
            scale_b,
            M, N, K);
    }

    HIP_API_CALL(hipGetLastError());
    HIP_API_CALL(hipDeviceSynchronize());
}

void verifyCPU()
{
    constexpr int MS = 288;
    constexpr int NS = 576;
    constexpr int KS = 256;

    Matrix<float8> a(KS, MS);
    Matrix<float8> b(KS, NS);
    Matrix<float> a32(KS, MS);
    Matrix<float> b32(KS, NS);
    Matrix<float16> c(MS, NS);
    Matrix<float> scale_a((KS+127)/128, MS);
    Matrix<float> scale_b((KS+127)/128, (NS+127)/128);

    for (int j=0; j<KS; j++)
    for (int i=0; i<MS; i++)
        a.host[j*MS+i] = (rand() % 4096)/4096.0f - 0.5f;

    for (int j=0; j<KS; j++)
    for (int i=0; i<NS; i++)
        b.host[j*NS+i] = (rand() % 4096)/4096.0f - 0.5f;

    for (int j=0; j<KS; j++)
    for (int i=0; i<MS; i++)
        a32.host[j*MS+i] = a.host[j*MS+i];

    for (int j=0; j<KS; j++)
    for (int i=0; i<NS; i++)
        b32.host[j*NS+i] = b.host[j*NS+i];

    for (int j=0; j<MS; j++)
    for (int i=0; i<KS/128; i++)
        scale_a.host[j*(KS/128) + i] = (rand() % 4096)/2048.0f - 1.0f;

    for (int j=0; j<KS/128; j++)
    for (int i=0; i<NS/128; i++)
        scale_b.host[j*(NS/128) + i] = (rand() % 4096)/2048.0f - 1.0f;

    a.toDevice();
    b.toDevice();
    scale_a.toDevice();
    scale_b.toDevice();

    launchHip(a.dev, b.dev, c.dev, scale_a.dev, scale_b.dev, MS, NS, KS);

    c.toHost();

    int errcount = 0;
    float meanerr = 0;

    for (int j=0; j<NS; j++)
    for (int i=0; i<MS; i++)
    {
        float acc = 0;
        for (int k=0; k<KS; k++)
            acc += a32.host[k*MS+i] * b32.host[k*NS+j] * scale_a.host[(k/128)*MS + i] * scale_b.host[(k/128)*((NS+127)/128) + (j/128)];

        float v0 = c.host[i*NS+j];

        float diff = std::abs(acc-v0)/std::max(std::abs(acc), 1E-3f);
        meanerr += diff*diff;
        if (diff > 0.02f) if (++errcount < 128)
            std::cout << j << " " << i << " diff " << diff << " -- " << acc << " " << v0 << std::endl;
    }

    if (errcount > 0) std::cout << "########## CPU Errors: " << errcount << std::endl;
    std::cout << "########## Mean error: " << sqrtf(meanerr/MS/NS) << std::endl;
}

void launchREF(
    const void* a,
    const void* b,
    void* c,
    const float* scale_a,
    const float* scale_b,
    int M, int N, int K
);

void verifyGPU(int MS, int NS, int KS)
{
    Matrix<float8> a(KS, MS);
    Matrix<float8> b(KS, NS);
    Matrix<float16> c(MS, NS);
    Matrix<float16> c_ref(MS, NS);
    Matrix<float> scale_a((KS+127)/128, MS);
    Matrix<float> scale_b((KS+127)/128, (NS+127)/128);

    for (int j=0; j<KS; j++)
    for (int i=0; i<MS; i++)
        a.host[j*MS+i] = (rand() % 8192)/1024.0f - 4.0f;

    for (int j=0; j<KS; j++)
    for (int i=0; i<NS; i++)
        b.host[j*NS+i] = (rand() % 8192)/1024.0f - 4.0f;

    for (int j=0; j<MS; j++)
    for (int i=0; i<KS/128; i++)
        scale_a.host[j*(KS/128) + i] = (rand() % 8192)/1024.0f - 4.0f;

    for (int j=0; j<KS/128; j++)
    for (int i=0; i<NS/128; i++)
        scale_b.host[j*(NS/128) + i] = (rand() % 8192)/1024.0f - 4.0f;

    a.toDevice();
    b.toDevice();
    scale_a.toDevice();
    scale_b.toDevice();

    launchREF(a.dev, b.dev, c_ref.dev, scale_a.dev, scale_b.dev, MS, NS, KS);
    launchHip(a.dev, b.dev, c.dev, scale_a.dev, scale_b.dev, MS, NS, KS);

    c.toHost();
    c_ref.toHost();

    int errcount = 0;
    double meanerr = 0;

    for (int j=0; j<NS; j++)
    for (int i=0; i<MS; i++)
    {
        float acc = c_ref.host[i*NS+j];
        float v0 = c.host[i*NS+j];

        float diff = std::abs(acc-v0)/std::max(std::abs(acc), 1E-3f);
        meanerr += diff*diff;
        if (diff > 0.02f) if (++errcount < 128)
            std::cout << j << " " << i << " diff " << diff << " -- " << acc << " " << v0 << std::endl;
    }

    meanerr = sqrtf(meanerr/MS/NS);
    if (errcount > 0)
        std::cout << "########### " << MS << '.' << NS << '.' << KS << " Errors: " << errcount << std::endl;
    if (meanerr < 1E-5) // err<1e-5 is false for NaN, so compare on the opposite direction
        {}
    else
        std::cout << "Mean error: " << meanerr << std::endl;
}

float run(int M, int N, int K)
{
    Matrix<float8> a(K, M);
    Matrix<float8> b(K, N);
    Matrix<float16> c(M, N);
    Matrix<float> scale_a(K/128, M);
    Matrix<float> scale_b(K/128, N/128);

    for (int i=0; i<10; i++)
        launchHip(a.dev, b.dev, c.dev, scale_a.dev, scale_b.dev, M, N, K);

    HIP_API_CALL(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    int runs = 500;
    for (int i=0; i<runs; i++)
        launchHip(a.dev, b.dev, c.dev, scale_a.dev, scale_b.dev, M, N, K);

    HIP_API_CALL(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    float time = (t1-t0).count()/1000.0f/runs;
    std::cout << M << "." << N << "." << K << " " << time << std::endl; 

    return time;
}

struct testcase
{
    size_t m, n, k;
};

int main()
{
    hipDeviceProp_t devProp{};
    hipGetDeviceProperties(&devProp, 1);
    hipSetDevice(1);

    verifyCPU();

    /*std::vector<testcase> testcases = {
        {1024, 1536, 7168},
        {1024, 4608, 7168},
        {6144, 1536, 7168},
        {6144, 4608, 7168},
        {1024, 7168,  256},
        {6144, 7168,  256},
        {1024, 576,  7168}
    };*/
    std::vector<testcase> testcases = {
        {1024, 1536, 7168},
        {1024, 3072, 7168},
        {1024, 576, 7168},
        {1024, 4608, 7168},
        {1024, 512, 7168},
        {1024, 7168, 2048},
        {1024, 6144, 1536},
        {1024, 7168, 256},
        {1024, 7168, 2304},
        {1024, 4096, 512},
        {6144, 3072, 1536},
        {6144, 576, 7168},
        {6144, 7168, 256},
        {6144, 7168, 2048},
        {6144, 4608, 7168},
        {6144, 7168, 2304},
        {6144, 512, 7168},
        {6144, 4096, 512},
    };

    std::sort(testcases.begin(), testcases.end(), [](const testcase& a, const testcase& b) { return a.m*a.k*a.n > b.m*b.k*b.n; });

    for (auto& test : testcases)
    {
        int num = test.m * test.n / SHMBLOCK / SHMBLOCK;
        int mi300_cu = 304;
        std::cout << test.m << '.' << test.n << '.' << test.k << " bpcu: " << num / mi300_cu << " + " << (num % mi300_cu) << " N3=" << test.m*test.n*test.k/1024/1024 << std::endl;
        verifyGPU(test.m, test.n, test.k);
    }

    float time = 1.0f;
    for (auto& test : testcases) time *= powf(run(test.m, test.n, test.k), 1.0f/testcases.size());

    std::cout << "Time: " << 0.1*int(time*10) << std::endl;

    return 0;
}

template<int WIDTH>
__global__ void __launch_bounds__(TBLOCK*TBLOCK*2, 4)
reference_gemm(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    float16* __restrict__ c,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int MDIM, int NDIM, int KDIM
) {
    const int X = blockIdx.x * SHMBLOCK * WIDTH;
    const int Y = blockIdx.y * SHMBLOCK;
    
    const int TX = threadIdx.x%TBLOCK;
    const int TY = threadIdx.y%4;
    const int TZ = (threadIdx.y/4)%4;
    const int TYZ = threadIdx.y%16;
    const int TW = (threadIdx.y/16)%2;

    __shared__ uint8_t a_shared[WIDTH][4][4][4][TBLOCK][4];
    __shared__ uint8_t b_shared[4][4][4][TBLOCK][4];
    __shared__ float   scalar[4][TBLOCK][WIDTH];

    if (TW != 0)
    {
        for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            int k0 = k1 + k2;
            IVec4 temp_a[WIDTH], temp_b;

            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && Y + 4*TX < NDIM)
                temp_b[j1] = *reinterpret_cast<const uint32_t*>(&b[(k0 + j1*TBLOCK + TYZ)*NDIM + Y + 4*TX]);

            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && X < MDIM)
            for (int n=0; n<WIDTH; n++)
                temp_a[n][j1] = (X + n*SHMBLOCK < MDIM) ? *reinterpret_cast<const uint32_t*>(&a[(k0 + j1*TBLOCK + TYZ)*MDIM + X + 4*TX + n*SHMBLOCK]) : 0;

            int tx3 = (TX & 3) << 2;
            int i1 = TX >> 2;

            if (k0 != 0) __syncthreads();

            for (int j1=0; j1<4; j1++)
            for (int m=0; m<4; m++)
                b_shared[j1][i1][TY][tx3 + m][TZ] = (temp_b[j1] >> (8*m)) & 0xFF;

            for (int j1=0; j1<4; j1++)
            for (int m=0; m<4; m++)
            for (int n=0; n<WIDTH; n++)
                a_shared[n][i1][TZ][TY][tx3 + m][j1] = (temp_a[n][j1] >> (8*m)) & 0xFF;

            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
            __builtin_amdgcn_s_barrier();
        }
        return;
    }

    Vec16 reg_res[WIDTH];
    for (int n=0; n<WIDTH; n++) reg_res[n] = {};

    float tmp_a[WIDTH];
    float tmp_b;
    if (TZ == 0)
    {
        for (int n=0; n<WIDTH; n++)
            tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
        tmp_b = scale_b[Y/128];
    }

    for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
    {
        if (TZ == 0)
        {
            for (int n=0; n<WIDTH; n++) scalar[TY][TX][n] = tmp_b * tmp_a[n];

            if (k1 + 2*SHMBLOCK < KDIM)
            {
                for (int n=0; n<WIDTH; n++)
                    tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[(k1/128 + 1)*MDIM + X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
                tmp_b = scale_b[(k1/128 + 1)*((NDIM+127)/128) + (Y/128)];
            }
        }
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            int k0 = k1 + k2;

            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
            
            float8x8 a0_load[WIDTH], a1_load[WIDTH];
            float8x8 b0_load[4], b1_load[4];

    #pragma unroll
            for (int m=0; m<8; m++)
    #pragma unroll
            for (int n=0; n<4; n++)
            {
                b0_load[n][m] = b_shared[m/4 + 0][n][TY][TX][m%4];
                b1_load[n][m] = b_shared[m/4 + 2][n][TY][TX][m%4];
            }

    #pragma unroll
            for (int m=0; m<8; m++)
    #pragma unroll
            for (int r=0; r<WIDTH; r++)
            {
                a0_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 0];
                a1_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 2];
            }

            Vec4 scal[WIDTH];
    #pragma unroll
            for (int n=0; n<WIDTH; n++)
    #pragma unroll
            for (int m=0; m<4; m++)
                scal[n][m] = scalar[TZ][TY*4 + m][n];

            __syncthreads();

    #pragma unroll
            for (int n=0; n<4; n++)
    #pragma unroll
            for (int r1=0; r1<WIDTH; r1++)
            {
                Vec4 res = mfma(a0_load[r1], a1_load[r1], b0_load[n], b1_load[n]);

                for (int m=0; m<4; m++)
                    reg_res[r1][m*4 + n] += scal[r1][m] * res[m];
            }
        }
    }

    for (int j=0; j<4; j ++) for (int i=0; i<4; i ++) for (int r1=0; r1<WIDTH; r1++)
    if (Y + i*TBLOCK < NDIM && 4*TYZ + X + r1*SHMBLOCK < MDIM)
        c[(4*TYZ + j + X + r1*SHMBLOCK)*NDIM + Y + i*TBLOCK + TX] = (float16) reg_res[r1][j*4 + i];
}

void launchREF(
    const void* a,
    const void* b,
    void* c,
    const float* scale_a,
    const float* scale_b,
    int M, int N, int K
) {
    dim3 block(TBLOCK, 32, 1);

    int mi300_threads = SHMBLOCK * SHMBLOCK * 304;
    int bpcu = M * N / mi300_threads;

    if (bpcu >= 6 && M > 3*SHMBLOCK)
    {
        dim3 grid((M + 4*SHMBLOCK - 1) / SHMBLOCK / 4, N / SHMBLOCK);

        reference_gemm<4><<<grid, block, 0, 0>>>(
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<float16*>(c),
            scale_a,
            scale_b,
            M, N, K);
    }
    else if (bpcu >= 2 && M > SHMBLOCK)
    {
        dim3 grid((M + 2*SHMBLOCK - 1) / SHMBLOCK / 2, N / SHMBLOCK);

        reference_gemm<2><<<grid, block, 0, 0>>>(
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<float16*>(c),
            scale_a,
            scale_b,
            M, N, K);
    }
    else
    {
        dim3 grid((M + SHMBLOCK - 1) / SHMBLOCK, N / SHMBLOCK);

        reference_gemm<1><<<grid, block, 0, 0>>>(
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b),
            reinterpret_cast<float16*>(c),
            scale_a,
            scale_b,
            M, N, K);
    }

    hipDeviceSynchronize();
}