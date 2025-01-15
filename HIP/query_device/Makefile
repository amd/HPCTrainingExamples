# ------------------------------

HIPCC    = hipcc
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))
HIPFLAGS = --offload-arch=${ROCM_GPU}

# ------------------------------

query_device: query_device.o
	${HIPCC} ${HIPFLAGS} query_device.o -o query_device

query_device.o: query_device.cpp
	${HIPCC} ${HIPFLAGS} -c query_device.cpp

.PHONY: clean

clean:
	rm -f query_device *.o
