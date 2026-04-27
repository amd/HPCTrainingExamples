.PHONY: build clean test

build:
	python setup.py build_ext --inplace

clean:
	rm -f compute.c hpc_kernels.o compute*.so compute*.pyd
	rm -rf build *.egg-info

test: build
	python driver.py
