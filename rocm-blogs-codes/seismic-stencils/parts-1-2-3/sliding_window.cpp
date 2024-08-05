#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <assert.h>

#include "helper.hpp"
#include "fd_coefficients.hpp"

template <int R>
__inline__ void compute_fd_sliding_window(float *p_out, const float *p_in, const float *d, int begin, int end, int pos, int stride) {
    /* Computes a central high order finite difference approximation in a given direction using a sliding window
     p_out: Array to write the result to
     p_in: Array to apply the approximation to
     d: Array of length 2 * R + 1 that contains the finite difference approximations, including scaling by grid spacing. 
     R: Stencil radius
     begin: The index to start the sliding window from
     end: The index to stop the sliding at (exclusive)
     pos: The current grid point to apply the approximation to
     stride: The stride to use for the stencil. This parameter controls the direction in which the stencil is applied.
    */
     
    // Sliding window
    float w[2 * R + 1];
     
    // 1. Prime the sliding window
    for (int r = 0; r < 2 * R; r++)
        w[r] = p_in[pos + (begin - R + r) * stride]; 
    
    // Apply the sliding window along the given grid direction determined by `stride`
    for (int i = begin; i < end; ++i) {
        // 2. Load the next value into the sliding window at its last position
        w[2 * R] = p_in[pos + (i + R) * stride];
        // 3. Compute the finite difference approximation using the sliding window
        float out = 0.0f;
        for (int r = 0; r <= 2 * R; r++)
            out += w[r] * d[r]; 
        p_out[pos + i * stride] = out;
        // 4. Update the sliding window by shifting it forward one step
        for (int r = 0; r < 2 * R; r++)
            w[r] = w[r+1];
        
    }
}

// Compute the maximum absolute value of the difference of u - v
float max_absval(float *u, float *v, int line, int slice, int x0, int x1, int y0, int y1, int z0, int z1) {

    float maxval = 0.0f;
    for (int k = z0; k < z1; ++k)
    for (int j = y0; j < y1; ++j)
    for (int i = x0; i < x1; ++i) {
        size_t pos = POS(i, j, k);
        float diff = fabs(u[pos] - v[pos]);
        maxval = diff > maxval ? diff : maxval;
    }

    return maxval;
}

int main(int argc, char **argv) {

    // Default grid size
    int nx = 10;
    int ny = 10;
    int nz = 10;

    // Read grid size from command line (if provided)
    if (argc > 1) 
        nx = atof(argv[1]);
    if (argc > 2) 
        ny = atof(argv[2]);
    if (argc > 3) 
        nz = atof(argv[3]);


    // Finite difference approximation to use
    const int R = 1;
    const float h = 1.0f;
    float d[2*R + 1];
    fd_coefficients_d2_order_2(d, R, h);

    // Padded grid size
    const int mx = nx + 2 * R;
    const int my = ny + 2 * R;
    const int mz = nz + 2 * R;

    const int line = mx;
    const int slice = mx * my;

    // Total number of grid points
    const size_t m = (size_t)mx * (size_t)my * (size_t)mz;

    // Define each direction to apply the finite difference approximation in
    const int x = 1;
    const int y = line;
    const int z = slice;

    float *p_in = new float[m];
    float *p_out = new float[m];
    float *p_ref = new float[m];

    // Initialize input, output, and reference arrays
    for (int k = 0; k < mz; ++k)
    for (int j = 0; j < my; ++j)
    for (int i = 0; i < mx; ++i) {
        size_t pos = POS(i, j, k);
        assert(pos < m);
        p_in[pos] = i + j + k;
        p_out[pos] = 2.0f;
        p_ref[pos] = 0.0f;
    }

    // Apply sliding window in the z-direction
    for (int j = R; j < ny + R; ++j) 
    for (int i = R; i < nx + R; ++i) {
        size_t pos = POS(i, j, 0);
        compute_fd_sliding_window<R>(p_out, p_in, d, R, nz + R, pos, z);
    }

    // Check result
    float maxval = max_absval(p_out, p_ref, line, slice, R, nx + R, R, ny + R, R, nz + R);
    printf("Maximum absolute pointwise difference: %g \n", maxval);

    delete[] p_in;
    delete[] p_out;
    delete[] p_ref;


}
