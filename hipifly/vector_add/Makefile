#-- make.host for AMD Cloud Platform (ACP)
#-- Compiler and flags for different build type

# Compiler for host code
CXX               = g++

# nvcc is the default compiler for the device code
GPUCXX            ?= nvcc

# Uncomment this to use HIP
# DFLAGS += -DENABLE_HIP  

CFLAGS   = -g -O2
CXXFLAGS = -Ofast -std=c++11
ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

DIRS     := src 
CFILES   := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.c))
CPPFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cpp))
GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))

# Build a list of all potential object files so cleaning works properly
CLEAN_OBJS := $(subst .c,.o,$(CFILES)) \
              $(subst .cpp,.o,$(CPPFILES)) \
              $(subst .cu,.o,$(GPUFILES))

OBJS     := $(subst .c,.o,$(CFILES)) \
            $(subst .cpp,.o,$(CPPFILES)) \
            $(subst .cu,.o,$(GPUFILES))

CFLAGS   ?= -Ofast
CXXFLAGS ?= -Ofast -std=c++11
GPUFLAGS ?= -O3 -std=c++11

#-- Add flags and libraries as needed
CFLAGS   += $(DFLAGS) -Isrc
CXXFLAGS += $(DFLAGS) -Isrc
GPUFLAGS += $(DFLAGS) -Isrc

 
# Use appropiate flags whether using CUDA or HIP
ifeq ($(findstring -DENABLE_HIP,$(DFLAGS)),-DENABLE_HIP)
# Set hipcc as the compiler for device code
	GPUCXX    = hipcc
	CXXFLAGS  += -I$(ROCM_PATH)/include
	GPUFLAGS  += -O3 --offload-arch=$(ROCM_GPU)
	LIBS      += -L$(ROCM_PATH)/lib -lamdhip64 
else
	CXXFLAGS  += -I$(CUDA_PATH)/include
	GPUFLAGS  += -O3 -arch=sm_80
	LIBS      += -L$(CUDA_PATH)/lib64 -lcudart
endif		

LD        := $(CXX)
LDFLAGS   := $(CXXFLAGS)

.SUFFIXES: .cpp .cu .o

EXEC := vector_add

$(EXEC): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $(EXEC) $(LIBS)
	
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(GPUCXX) $(GPUFLAGS) -c $< -o $@ 

.PHONY: clean

clean:
	rm -f $(CLEAN_OBJS)
	rm -f $(EXEC)
