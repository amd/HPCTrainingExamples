#!/bin/bash

# This test runs an example from https://github.com/CliMA/Oceananigans.jl

# NOTE: this test assumes Julia has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/julia_setup.sh

module load julia

julia -e 'using Pkg; Pkg.add("MPI")'
julia -e 'using Pkg; Pkg.add("Oceananigans")'
julia -e 'using Pkg; Pkg.add("CUDA")'
julia -e 'using Pkg; Pkg.add("FFTW")'
julia -e 'using Pkg; Pkg.add("KernelAbstractions")'
julia -e 'using Pkg; Pkg.add("SeawaterPolynomials")'
julia -e 'using Pkg; Pkg.add("OffsetArrays")'
julia -e 'using Pkg; Pkg.add("JLD2")'
julia -e 'using Pkg; Pkg.add("Adapt")'
julia -e 'using Pkg; Pkg.add("GPUArraysCore")'

git clone https://github.com/CliMA/Oceananigans.jl.git
pushd Oceananigans.jl/test
julia test_amdgpu.jl
popd
rm -rf Oceananigans.jl
