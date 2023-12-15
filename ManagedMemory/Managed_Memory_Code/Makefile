all: gpu_code

gpu_code: gpu_code.hip
	hipcc -g -O3 -fopenmp --offload-arch=native gpu_code.hip -o gpu_code

clean:
	rm -rf gpu_code
