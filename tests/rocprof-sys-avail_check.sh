#!/bin/bash

# This test checks that the rocprof-sys-avail
# (formerly) omnitrace-avail
# binary exists and it is able to write
# a config file

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if command -v rocprof-sys-avail &> /dev/null; then
    echo "rocprof-sys-avail found at: $(which rocprof-sys-avail)"
else
    echo "loading rocprofiler-systems module"
    module load rocprofiler-systems
fi

rocprof-sys-avail -G $PWD/.configure.cfg

rm .configure.cfg
