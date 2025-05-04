EXEC = ShallowWater
default: ${EXEC}
all: ${EXEC}

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

# hipstdpar-path is not needed for ROCm 6.1 and later --hipstdpar-path=${STDPAR_PATH}
ifneq ($(findstring amdclang++,$(CXX1)),)
  STDPAR_FLAGS = --hipstdpar --offload-arch=$(ROCM_GPU) --hipstdpar-path=${STDPAR_PATH}
  #STDPAR_FLAGS = --hipstdpar --hipstdpar-path=${STDPAR_PATH}
else ifneq ($(findstring clang++,$(CXX1)),)
  STDPAR_FLAGS = --hipstdpar --offload-arch=$(ROCM_GPU) --hipstdpar-path=${STDPAR_PATH}
  #STDPAR_FLAGS = --hipstdpar --hipstdpar-path=${STDPAR_PATH}
endif
# Add --hipstdpar-interpose-alloc if HSA_XNACK is not set
ifneq ($(findstring gfx1030,$(ROCM_GPU)),)
  STDPAR_FLAGS += --hipstdpar-interpose-alloc
endif

CXXFLAGS = -g -O3 -fstrict-aliasing ${STDPAR_FLAGS}
LDFLAGS = -fno-lto -lm

ShallowWater.o: ShallowWater.cpp Var2D.hpp Makefile

${EXEC}: ${EXEC}.o
	${CXX} ${STDPAR_FLAGS} $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o ${EXEC}
