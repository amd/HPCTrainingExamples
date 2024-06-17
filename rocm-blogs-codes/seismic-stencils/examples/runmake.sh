#!/bin/sh
for VEC in 0 1 2
do
  for RADIUS in 1 2 3 4
  do
    make vec=$VEC radius=$RADIUS
  done
done
