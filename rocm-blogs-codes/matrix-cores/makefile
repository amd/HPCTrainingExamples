#####################################################################################
#
#Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
#
#####################################################################################

define HELP_MSG

MFMA Examples makefile targets:

	 make all (default)
	 make clean
	 make info
	 make help

Usage:

make all
	 Build MFMA examples
make clean
	 Clean all binaries and object files.
make info
	 List directories and compiler flags in use.
make help
	 Display this help message.

endef

ifeq (,$(filter all clean info help, $(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${HELP_MSG})
endif
endif

MFMA_DIR:=$(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
INCLUDE_DIR=${MFMA_DIR}/include

#compiler
HIPCC=hipcc

#includes
INCLUDES=-I${INCLUDE_DIR}

#defines
DEFINES=

#.cpp compilation flags
CXXFLAGS=-O3 --offload-arch=gfx90a,gfx908

#libraries
LIBS=

#object dependencies
DEPS=$(wildcard ${INCLUDE_DIR}/*.hpp)

#source files
SRC =$(wildcard src/*.cpp)

#binaries
BINS=$(SRC:src/%.cpp=%)

.PHONY: all clean

all: $(BINS)

# rule for .cpp files
%: src/%.cpp $(DEPS)
	$(HIPCC) $(DEFINES) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

#cleanup
clean:
	rm -f $(BINS)

help:
	$(info $(value HELP_MSG))
	@true

info:
	$(info MFMA_DIR  = $(MFMA_DIR))
	$(info CXXFLAGS  = $(CXXFLAGS))
	$(info DEFINES   = $(DEFINES))
	$(info INCLUDES  = $(INCLUDES))
	$(info LIBS      = $(LIBS))
	$(info BINS      = $(BINS))
	@true
