#!/bin/bash

ls -l /opt/rocmplus-* | grep omnitrace

module load rocm omnitrace/1.11.2
module list

omnitrace-instrument --version
