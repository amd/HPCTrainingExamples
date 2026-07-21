# Vector Addition on AMD GPU with Julia

## Setup

First, make sure that you have the necessary dependencies installed, which are
- **Julia 1.12+** (required for MI300A/MI300X) - see [Julia Downloads](https://julialang.org/downloads/)
- ROCm installed on the system

If you are working on AAC6 or AAC7, make sure that a ROCm module is loaded with
```bash
module load rocm
```
Next, move to this directory, install the third-party `AMDGPU.jl` package and verify your GPU is detected:

```bash
cd Julia/vec_add
julia --project -e 'using Pkg; Pkg.instantiate(); using AMDGPU; AMDGPU.versioninfo()'
```

## Run

With Julia and the necessary packages installed, you can now run the script with

```bash
julia --project vec_add.jl
```

If the scripts runs succesfully, it should output: `PASS!`
