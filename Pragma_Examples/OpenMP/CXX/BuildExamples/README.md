# OpenMP CXX Build systems: make and cmake

README.md in `HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/BuildExamples` of the Training Examples repository

Build systems for make and cmake are an important starting step to working with OpenMP.
We'll show samples for CXX builds. We'll test them with
some of our sample code to make sure your system is setup properly.

## CXX

### Make

```
cd ../../CXX/BuildExamples
```

First let's take a look at the makefile

```
cat Makefile
```

The output should be

```
all: openmp_code

ROCM_GPU ?= $(strip $(shell rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *//'))

CXX1=$(notdir $(CXX))

ifeq ($(findstring amdclang,$(CXX1)), amdclang)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
else ifeq ($(findstring clang,$(CXX1)), clang)
  OPENMP_FLAGS = -fopenmp --offload-arch=${ROCM_GPU}
else ifeq ($(findstring gcc,$(CXX1)), gcc)
  OPENMP_FLAGS = -fopenmp -foffload=-march=${ROCM_GPU}
else ifeq ($(findstring CC,$(CXX1)), CC)
  OPENMP_FLAGS = -fopenmp
endif

CXXFLAGS = -g -O3 -fstrict-aliasing ${OPENMP_FLAGS}
LDFLAGS = ${OPENMP_FLAGS} -fno-lto -lm

openmp_code: openmp_code.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Cleanup
clean:
	rm -f *.o openmp_code
	rm -rf build
```

```
module load amdclang
make
```

Now run the executable

```
./openmp_code
```

### cmake

Looking at the CMakeLists.txt

```
cat CMakeLists.txt
```

The output should be

```
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(Memory_pragmas LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_BUILD_TYPE)

execute_process(COMMAND rocminfo COMMAND grep -m 1 -E gfx[^0]{1} COMMAND sed -e "s/ *Name: *//" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROCM_GPU)

string(REPLACE -O2 -O3 CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set(CMAKE_CXX_FLAGS "-fstrict-aliasing -faligned-allocation -fnew-alignment=256")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp --offload-arch=${ROCM_GPU}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -foffload=-march=${ROCM_GPU}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Cray")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
   #the cray compiler decides the offload-arch by loading appropriate modules
   #module load craype-accel-amd-gfx942 for example
endif()

add_executable(openmp_code openmp_code.cc)
```

```
module load amdclang
mkdir build && cd build && cmake ..
make
```

Now run the executable

```
./openmp_code
```

