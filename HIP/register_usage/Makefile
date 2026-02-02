EXECUTABLE = ./register_usage
all: clean $(EXECUTABLE)

.PHONY: test

OBJECTS = register_usage.o

HIPCC= $(shell which hipcc)
CXXFLAGS = -g -DNDEBUG -fPIC
HIPCC_FLAGS = -g -O3
SGPR ?= 1
VGPR ?= 0
ifeq ($(VGPR),1)
    SGPR := 0
endif
HIPCC_FLAGS += -DVGPR=$(VGPR) -DSGPR=$(SGPR)

HIP_PLATFORM ?= amd

ifeq ($(HIP_PLATFORM), nvidia)
   ROCM_PATH ?= $(shell hipconfig --path)
   HIPCC_FLAGS += -x cu -I${ROCM_PATH}/include/
endif
ifeq ($(HIP_PLATFORM), amd)
   HIPCC_FLAGS += -x hip -munsafe-fp-atomics -Rpass-analysis=kernel-resource-usage
endif

%.o: %.hip
	${HIPCC} $(HIPCC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS)
	${HIPCC} $< $(LDFLAGS) -o $@

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS)
