HIP_PATH?= $(wildcard ${ROCM_PATH}/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=${ROCM_PATH}
endif
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

.PHONY: test

all: vectoradd_hip.exe vectoradd_hip1.exe vectoradd_hip2.exe vectoradd_hip3.exe vectoradd_hip4.exe test

CXXFLAGS =-g -O2 --offload-arch=$(ROCM_GPU)

CXX=$(HIPCC)


vectoradd_hip.exe: vectoradd_hip.o
	$(HIPCC) vectoradd_hip.o -o $@

vectoradd_hip1.exe: vectoradd_hip1.o
	$(HIPCC) vectoradd_hip1.o -o $@

vectoradd_hip2.exe: vectoradd_hip2.o
	$(HIPCC) vectoradd_hip2.o -o $@

vectoradd_hip3.exe: vectoradd_hip3.o
	$(HIPCC) vectoradd_hip3.o -o $@

vectoradd_hip4.exe: vectoradd_hip4.o
	$(HIPCC) vectoradd_hip4.o -o $@

test: vectoradd_hip.exe vectoradd_hip1.exe vectoradd_hip2.exe vectoradd_hip3.exe vectoradd_hip4.exe
	./vectoradd_hip.exe
	./vectoradd_hip1.exe
	./vectoradd_hip2.exe
	./vectoradd_hip3.exe
	./vectoradd_hip4.exe

clean:
	rm -f *.o *.exe
