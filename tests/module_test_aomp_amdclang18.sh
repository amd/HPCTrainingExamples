#!/bin/bash

ls -l /opt/rocmplus-* | grep aomp_18.0-0

module load aomp/amdclang-18.0
module list

amdclang --version
