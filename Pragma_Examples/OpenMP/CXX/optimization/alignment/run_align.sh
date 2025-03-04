#!/bin/bash

for blocksize in "64 128 256 512 1024"
do
   for alignment_size in "16 32 64 128 256"
   do
      ./align_$blocksize $alignment_size
   done
done
