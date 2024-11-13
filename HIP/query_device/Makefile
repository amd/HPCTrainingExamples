# ------------------------------

HIPCC    = hipcc
HIPFLAGS = --offload-arch=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 |sed 's/ //g'`

# ------------------------------

query_device: query_device.o
	${HIPCC} query_device.o -o query_device

query_device.o: query_device.cpp
	${HIPCC} -c query_device.cpp

.PHONY: clean

clean:
	rm -f query_device *.o
