#!/bin/bash

ls -l  /opt/rocmplux-* |grep omniperf-1.0.10

module load rocm omniperf/1.0.10
module list

omniperf --version
