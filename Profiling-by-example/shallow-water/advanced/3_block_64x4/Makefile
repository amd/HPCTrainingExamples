EXECUTABLE = ./shallow_mpi
all: $(EXECUTABLE)

.PHONY: test

OBJECTS = shallow_mpi.o

HIPCC ?= hipcc
MPICXX ?= mpicxx
NRANKS ?= 2

# hipcc has to be the compiler because the source contains device code, so the
# MPI include and link flags are taken from the wrapper and passed to it.
IS_OMPI := $(shell $(MPICXX) --showme:compile >/dev/null 2>&1 && echo yes)
ifeq ($(IS_OMPI), yes)
   MPI_CFLAGS  = $(shell $(MPICXX) --showme:compile)
   MPI_LDFLAGS = $(shell $(MPICXX) --showme:link)
else
   MPI_CFLAGS  = $(shell $(MPICXX) -compile_info | cut -d" " -f 2-)
   MPI_LDFLAGS = $(shell $(MPICXX) -link_info | cut -d" " -f 2-)
endif

# The ROCTx ranges in the source come from the rocprofiler-sdk marker library.
ROCM_PATH ?= $(shell hipconfig --rocmpath 2>/dev/null || echo /opt/rocm)
ROCTX_CFLAGS  = -I$(ROCM_PATH)/include
ROCTX_LDFLAGS = -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx

CXXFLAGS = -g -O2 -DNDEBUG -fPIC
HIPCC_FLAGS = -g -O2 -DNDEBUG $(MPI_CFLAGS) $(ROCTX_CFLAGS)
LDFLAGS += $(MPI_LDFLAGS) $(ROCTX_LDFLAGS)

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   ROCM_PATH ?= $(shell hipconfig --path)
   HIPCC_FLAGS += -x cu -I${ROCM_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics
endif

%.o: %.hip
	${HIPCC} $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	${HIPCC} $< $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	mpirun -n $(NRANKS) $(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) build
