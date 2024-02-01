#!/bin/bash

ls -l /opt/rocmplus-* | grep aomp_18.0-0

module load aomp/amdclang18
module list

amdclang --version
