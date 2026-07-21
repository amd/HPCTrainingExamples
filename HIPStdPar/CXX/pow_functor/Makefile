default: pow_functor_inline pow_functor_inline_always pow_functor_working
all: pow_functor_inline pow_functor_inline_always pow_functor_working

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

# hipstdpar-path is not needed for ROCm 6.1 and later --hipstdpar-path=${STDPAR_PATH}
ifneq ($(findstring amdclang++,$(CXX1)),)
  STDPAR_FLAGS = --hipstdpar --offload-arch=$(ROCM_GPU) --hipstdpar-path=${STDPAR_PATH}
else ifneq ($(findstring clang++,$(CXX1)),)
  STDPAR_FLAGS = --hipstdpar --offload-arch=$(ROCM_GPU) --hipstdpar-path=${STDPAR_PATH}
endif
# Add --hipstdpar-interpose-alloc if HSA_XNACK is not set
ifneq ($(findstring gfx1030,$(ROCM_GPU)),)
  STDPAR_FLAGS += --hipstdpar-interpose-alloc
endif

CXXFLAGS = -g -O3 -std=c++20 -fstrict-aliasing ${STDPAR_FLAGS}
LDFLAGS = -fno-lto -lm


pow_functor_inline: pow_functor_inline.o
	${CXX} ${STDPAR_FLAGS} $(LDFLAGS) $^ -o $@

pow_functor_inline_always: pow_functor_inline_always.o
	${CXX} ${STDPAR_FLAGS} $(LDFLAGS) $^ -o $@

pow_functor_working: pow_functor_working.o
	${CXX} ${STDPAR_FLAGS} $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o pow_functor_inline pow_functor_inline_always pow_functor_working
