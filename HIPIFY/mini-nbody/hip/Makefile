all: nbody-orig nbody-soa nbody-block nbody-unroll

CXXFLAGS =-g -O2
HIPCC_FLAGS=-O2 -g -DNDEBUG
HIPCC_FLAGS+=-I.. -DSHMOO

HIPCC?=hipcc

ifeq ($(HIPCC), nvcc)
   HIPCC_FLAGS+=$(HIPCC_OPTS) -x cu
   LDFLAGS=-lcudadevrt -lcudart_static -lrt -lpthread -ldl
endif
ifeq ($(HIPCC), hipcc)
   HIPCC_FLAGS+=$(HIPCC_OPTS) -munsafe-fp-atomics
   ifeq (, $(findstring CXX, clang++))
      LDFLAGS=--hip-link
   else
      LDFLAGS=-L${ROCM_PATH}/hip/lib -lamdhip64
   endif
endif

%.hip: ../cuda/%.cu
	hipify-perl $^ > $@

%.o: %.hip
	$(HIPCC) $(HIPCC_FLAGS) -c $< -o $@

nbody-orig: nbody-orig.o
	$(HIPCC) $^ $(LDFLAGS) -o $@

nbody-soa: nbody-soa.o
	$(HIPCC) $^ $(LDFLAGS) -o $@

nbody-block: nbody-block.o
	$(HIPCC) $^ $(LDFLAGS) -o $@

nbody-unroll: nbody-unroll.o
	$(HIPCC) $^ $(LDFLAGS) -o $@

clean:
	rm -f nbody-orig nbody-soa nbody-block nbody-unroll
	rm -f nbody-orig.o nbody-soa.o nbody-block.o nbody-unroll.o
	rm -f results.copy_stats.csv results.csv results.db results.hip_stats.csv
	rm -f results.hsa_stats.csv results.json results.stats.csv results.sysinfo.txt
