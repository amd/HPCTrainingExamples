// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <stdio.h>

int main(void)
{
    double da[3] = { 1.5, 2.5, 3.5 };    /* 8 bytes each */
    float  fa[3] = { 1.5f, 2.5f, 3.5f }; /* 4 bytes each */
    double scale = 2.0;

    printf("before: scale = %g\n", scale);
    scale = scale * 10.0;                /* Ch 5: "jump" OVER this line */
    printf("after:  scale = %g\n", scale);

    printf("da = %g %g %g   fa = %g %g %g\n",
           da[0], da[1], da[2], fa[0], fa[1], fa[2]);
    return 0;
}
