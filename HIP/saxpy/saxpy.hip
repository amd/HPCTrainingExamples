/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_runtime.h>

__constant__ float a = 1.0f;
const float a_h = 1.0f;

void init (int n, float *x, float *y)
{
    for (std::size_t i = 0; i < n; ++i)
    {
	x[i] = 1.0;
	y[i] = 0.0;
    }
}

float test_accuracy(int n, float *y, float *y0)
{
    float err = 0.;
    for (std::size_t i = 0; i < n; ++i)
	err += (y[i] - y0[i])*(y[i] - y0[i]);
    err = std::sqrt(err)/n;
    //printf("l2norm(Error): %12.5e\n", err);
    return err;
}

void saxpy_cpu (int n, float const* x, int incx, float* y, int incy)
{
    for (std::size_t i = 0; i < n; ++i)
	y[i] += a_h*x[i];
}

__global__
void saxpy (int n, float const* x, int incx, float* y, int incy)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        y[i] += a*x[i];
}

int main()
{
    int n = 256;
    std::size_t size = sizeof(float)*n;

    float *h_x = new float [n];
    float *h_y = new float [n];
    init(n, h_x, h_y);

    float* d_x;
    float *d_y;
    hipMalloc(&d_x, size);
    hipMalloc(&d_y, size);
    hipMemcpy(d_x, h_x, size, hipMemcpyHostToDevice);
    hipMemcpy(d_y, h_y, size, hipMemcpyHostToDevice);

    int num_groups = 2;
    int group_size = 128;
    saxpy<<<num_groups, group_size>>>(n, d_x, 1, d_y, 1);
    hipDeviceSynchronize();
    hipMemcpy(h_y, d_y, size, hipMemcpyDeviceToHost);

    // Accuracy test
    float tol=1.e-12;
    float *h_x0 = new float [n];
    float *h_y0 = new float [n];
    init(n, h_x0, h_y0);
    saxpy_cpu(n, h_x0, 1, h_y0, 1);
    float err = test_accuracy(n, h_y, h_y0);
    if (err <= tol)
	printf("PASSED!\n");
    else
	printf("FAILED!\n");

    delete [] h_x;
    delete [] h_y;
    delete [] h_x0;
    delete [] h_y0;
    hipFree(d_x);
    hipFree(d_y);

    return 0;
}
