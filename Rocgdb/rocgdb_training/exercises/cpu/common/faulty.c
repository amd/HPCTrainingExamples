// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <stdio.h>

static int deref(int *p)
{
    return *p;              /* crashes when p is NULL */
}

int main(void)
{
    int *p = NULL;
    int x = deref(p);
    printf("%d\n", x);
    return 0;
}
