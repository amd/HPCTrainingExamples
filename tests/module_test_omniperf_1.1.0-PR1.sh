#!/bin/bash

ls -l /opt/rocmplus-* | grep omniperf-1.1.0-PR1

module load rocm omniperf/1.1.0-PR1
module list

omniperf --version
