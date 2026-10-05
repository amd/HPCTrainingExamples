// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <stdio.h>

/* Built at -O2, breakpoints on the assignments below all bind to ONE
   address: the compiler fuses these lines into a single instruction.
   Rebuild at -O0 (or -Og) to make each line addressable again. */
__attribute__((noinline)) int compute(int x)
{
    int a = 7;
    a = a + 7;
    int b = a * 2;
    int c = b + x;
    return c;
}

int main(int argc, char **argv)
{
    printf("%d\n", compute(argc));   /* argc keeps the result from folding away */
    return 0;
}
