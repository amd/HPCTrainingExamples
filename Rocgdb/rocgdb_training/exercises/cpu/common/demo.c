// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* A tiny helper: something to step INTO and to see on the stack. */
static int add(int a, int b)
{
    int s = a + b;
    return s;
}

/* Sum data[0..n-1].  The loop is the target for break / ignore / watch /
   conditional / display (Ch 4, Ch 8) and for "until" (Ch 3). */
static long sum_array(const int *data, int n)
{
    long total = 0;
    for (int i = 0; i < n; i++)
        total = add(total, data[i]);
    if (total < 0)                 /* never taken - see the "until" exercise */
        printf("overflow!\n");
    return total;
}

int main(int argc, char **argv)
{
    int data[8] = { 3, 1, 4, 1, 5, 9, 2, 6 };
    unsigned long namelen = strlen(argv[0]);   /* a library call (Ch 3) */
    long total = sum_array(data, 8);           /* next OVER / step INTO */
    printf("sum = %ld, name length = %lu\n", total, namelen);

    /* Long idle loop for the ATTACH exercise (Ch 2): start it with
       ./demo --spin &  in the background, then attach and inspect k. */
    if (argc > 1 && strcmp(argv[1], "--spin") == 0) {
        long k = 0;
        for (;;) {
            usleep(100000);        /* 0.1 s per tick */
            k++;
        }
    }
    return 0;
}
