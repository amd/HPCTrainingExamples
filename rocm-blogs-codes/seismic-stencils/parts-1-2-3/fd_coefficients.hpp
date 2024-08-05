#pragma once
#include <assert.h>

// Finite difference approximations of the second derivative
void fd_coefficients_d2_order_2(float *dout, int R, float h) {
    assert(R == 1);
    float scaling = 1.0 / (h * h);
    dout[0] =  1.0f  * scaling;
    dout[1] = -2.0f * scaling;
    dout[2] = dout[0];
}

void fd_coefficients_d2_order_4(float *dout, int R, float h) {
    assert(R == 2);
    float scaling = 1.0 / (h * h);
    dout[0] = -(float)(1.0 / 12.0) * scaling;
    dout[1] =  (float)(4.0 / 3) * scaling;
    dout[2] = -(float)(5.0 / 2) * scaling;
    dout[3] = dout[1];
    dout[4] = dout[0];
}

void fd_coefficients_d2_order_6(float *dout, int R, float h) {
    assert(R == 3);
    float scaling = 1.0 / (h * h);
    dout[0] =  (float)(1.0 / 90.0) * scaling;
    dout[1] = -(float)(3.0 / 20.0) * scaling;
    dout[2] =  (float)(3.0 / 2.0) * scaling;
    dout[3] = -(float)(49.0 / 18.0) * scaling;
    dout[4] = dout[2];
    dout[5] = dout[1];
    dout[6] = dout[0];
}

void fd_coefficients_d2_order_8(float *dout, int R, float h) {
    assert(R == 4);
    float scaling = 1.0 / (h * h);
    dout[0] = -(float)(1.0 / 560.0) * scaling;
    dout[1] =  (float)(8.0 / 315.0) * scaling;
    dout[2] = -(float)(1.0 / 5.0) * scaling;
    dout[3] =  (float)(8.0 / 5.0) * scaling;
    dout[4] = -(float)(205.0 / 72.0) * scaling;
    dout[5] = dout[3];
    dout[6] = dout[2];
    dout[7] = dout[1];
    dout[8] = dout[0];
}
