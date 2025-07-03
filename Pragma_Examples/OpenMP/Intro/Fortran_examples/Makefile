all:target_data_structured target_data_unstructured target_data_update

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

OPENMP_FLAGS = -fopenmp
OPENMP_OFFLOAD_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}

FFLAGS = -g -O3 ${FREE_FORM_FLAG}

target_data_structured.o: target_data_structured.F90
	$(FC) -c $(FFLAGS) ${OPENMP_FLAGS} $^

target_data_structured: target_data_structured.o
	$(FC) $(LDFLAGS) ${OPENMP_FLAGS} $^ -o $@

target_data_unstructured.o: target_data_unstructured.F90
	$(FC) -c $(FFLAGS) ${OPENMP_FLAGS} $^

target_data_unstructured: target_data_unstructured.o
	$(FC) $(LDFLAGS) ${OPENMP_FLAGS} $^ -o $@

target_data_update.o: target_data_update.F90
	$(FC) -c $(FFLAGS) ${OPENMP_FLAGS} $^

target_data_update: target_data_update.o
	$(FC) $(LDFLAGS) ${OPENMP_FLAGS} $^ -o $@

# Cleanup
clean:
	rm -f *.o  *.mod target_data_structured target_data_unstructured target_data_update
