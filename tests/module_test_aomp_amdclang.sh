#!/bin/bash

ls -l /opt/rocmplus-* | grep aomp_19.0-0

module load aomp/amdclang-19.0
module list

$CC --version
