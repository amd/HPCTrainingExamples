###############################################################################
#Copyright (c) 2023 Advanced Micro Devices, Inc. (AMD)
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
###############################################################################

AMD_ARCH=gfx90a
ROCM_PATH ?=/opt/rocm
CC = ${ROCM_PATH}/bin/hipcc
CFLAGS = -O2 -g -std=c++17 --offload-arch=${AMD_ARCH} -I${ROCM_PATH}/include
LDFLAGS = -L${ROCM_PATH}/lib -lrocsparse

EXAMPLES = block_ellpack ellpack rocsparse_csr scalar_csr vector_csr

all: $(EXAMPLES)

%.o: %.cpp
	$(CC) $(CFLAGS) $(FLAGS) -o $@ -c $<

block_ellpack: block_ellpack.o
	$(CC) -o $@ $@.o

ellpack: ellpack.o
	$(CC) -o $@ $@.o

rocsparse_csr: rocsparse_csr.o
	$(CC) -o $@ $@.o $(LDFLAGS)

scalar_csr: scalar_csr.o
	$(CC) -o $@ $@.o

vector_csr: vector_csr.o
	$(CC) -o $@ $@.o

clean:
	rm -f ${EXAMPLES} *.o
