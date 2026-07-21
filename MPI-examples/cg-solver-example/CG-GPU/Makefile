# Makefile for the distributed GPU Conjugate Gradient solver
#
# Two supported toolchains:
#
#   OpenMPI + ROCm (e.g. AAC6 / MI300A Ubuntu):
#     module load rocm/6.4.1
#     module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4   # GPU-Aware MPI
#
#   HPE Cray EX + cray-mpich (PrgEnv-amd):
#     module swap PrgEnv-cray PrgEnv-amd/8.7.0    # AMD compilers + ROCm
#     module load craype-accel-amd-gfx942         # GPU-aware cray-mpich (GTL)
#   The Cray `CC` wrapper then supplies cray-mpich + the GPU-aware GTL library
#   (libmpi_gtl_hsa) automatically; only the ROCm math libraries are added here.
#
# Usage:
#   make                      # build ./cg_gpu (auto-detects the toolchain)
#   make GPU_ARCH=gfx90a      # build for a specific GPU architecture
#   make SPMV_V2=1            # use rocsparse_v2_spmv (ROCm 7.x); default uses rocsparse_spmv
#   make SCOREP=1             # instrument with Score-P (MPI + ROCm/HIP GPU regions)
#   make run                  # quick 4-rank sanity run over four methods
#   make clean

# ---------------------------------------------------------------------------
# ROCm location.  Under PrgEnv-amd the toolchain lives in a versioned prefix
# (e.g. /opt/rocm-7.0.3) exported as CRAY_AMD_COMPILER_PREFIX; otherwise the
# classic /opt/rocm symlink is used.
# ---------------------------------------------------------------------------
ifeq ($(origin ROCM_PATH), undefined)
  ifneq ($(CRAY_AMD_COMPILER_PREFIX),)
    ROCM_PATH := $(CRAY_AMD_COMPILER_PREFIX)
  else
    ROCM_PATH := /opt/rocm
  endif
endif

# Detect the local GPU architecture; fall back to MI300 (gfx942) when rocminfo
# is unavailable (e.g. building on a GPU-less login node).
GPU_ARCH ?= $(shell rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || echo gfx942)

# ---------------------------------------------------------------------------
# Compiler / MPI wrapper selection.
#   * Cray PE (PE_ENV set): use the `CC` wrapper with `-x hip`.  The wrapper
#     injects cray-mpich and, with craype-accel-amd-gfx942 loaded, the GPU-aware
#     GTL library, so no explicit -lmpi is needed here.
#   * Otherwise: use `mpicxx` with hipcc as the underlying compiler (OpenMPI).
# ---------------------------------------------------------------------------
ifdef PE_ENV
  MPICXX  ?= CC
  HIP_LANG = -x hip
else
  MPICXX  ?= mpicxx
  export OMPI_CXX = hipcc
  HIP_LANG =
endif

# ---------------------------------------------------------------------------
# Optional Score-P instrumentation (make SCOREP=1).
#   Wraps the MPI compiler with the Score-P measurement system.  We use
#   --nocompiler on purpose: automatic compiler instrumentation of the
#   HIP/rocSPARSE/rocBLAS header inlines generates so many regions that it
#   overflows the Score-P trace buffer.  --nocompiler still records the MPI
#   adapter and, at run time with SCOREP_ROCM_ENABLE=yes, the ROCm/HIP GPU
#   kernels and memcpys, which is exactly the communication+compute view we want.
#   GPU capture is validated on ROCm 6.4.x + scorep/9.4 (see docs/profilers/scorep.md).
# ---------------------------------------------------------------------------
ifeq ($(SCOREP),1)
  MPICXX := scorep --nocompiler --mpp=mpi $(MPICXX)
endif

CXXFLAGS = -O3 -std=c++17 $(HIP_LANG) --offload-arch=$(GPU_ARCH) -I$(ROCM_PATH)/include -Isrc
LDFLAGS  = -L$(ROCM_PATH)/lib
LIBS     = -lrocsparse -lrocblas -lrccl -lamdhip64

# Opt in to the ROCm 7.x rocsparse_v2_spmv path (persistent descriptor + one-time
# analysis stage).  Falls back to classic rocsparse_spmv automatically if the v2
# header is unavailable (e.g. ROCm 6.x).
ifeq ($(SPMV_V2),1)
CXXFLAGS += -DUSE_V2_SPMV
endif

TARGET  = cg_gpu
SRC     = src/cg.cpp
HEADERS = src/sparse_mat.hpp src/par_binary_IO.hpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) $(HEADERS)
	$(MPICXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LIBS)

# Quick sanity run: 4 ranks, four methods
run: $(TARGET)
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm isend
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm staged
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm alltoallv
	mpirun -n 4 ./$(TARGET) src/Dubcova2.pm rccl

clean:
	rm -f $(TARGET)
