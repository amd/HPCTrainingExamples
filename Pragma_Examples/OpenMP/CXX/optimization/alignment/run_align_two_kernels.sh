#!/bin/bash

for blocksize in 64 128 256 512 1024
do
   for alignment_size in 16 32 64 128 256
   do
      ./align_two_kernels_$blocksize $alignment_size
   done
done
