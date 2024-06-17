###############################################################################
#Copyright (c) 2022 Advanced Micro Devices, Inc. (AMD)
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

CC=hipcc
FLAGS=-O3 -std=c++17
#TEMPS=true
#THEORY=true
DOUBLE=true
EXAMPLE=kernel1 kernel2 kernel3 kernel4 kernel5 kernel6 kernel7

ifeq "$(DOUBLE)" "true"
	PRECISION=-DDOUBLE
	SUFFIX=dp
else
	PRECISION=
	SUFFIX=sp
endif

ifeq "$(TEMPS)" "true"
	FLAGS+=--save-temps
endif

ifeq "$(THEORY)" "true"
	FLAGS+=-DTHEORY
endif

all: $(EXAMPLE)

$(EXAMPLE):
	$(CC) -o laplacian_$(SUFFIX)_$@ laplacian.cpp $(FLAGS) $(PRECISION) -include $@.hpp
ifeq "$(TEMPS)" "true"
	mkdir -p temps
	mv *.s temps
	mv temps/*laplacian-hip* temps/laplacian_$(SUFFIX)_$@-hip.s
	mv temps/*laplacian-host* temps/laplacian_$(SUFFIX)_$@-host.s
	rm *-hip* *-host*
endif

clean:
	rm -rf temps laplacian_* *-hip* *-host*
