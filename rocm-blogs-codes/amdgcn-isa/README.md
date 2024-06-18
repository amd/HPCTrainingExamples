# Reading AMD GPU ISA

Code examples for the following blog:

[Reading AMD GPU ISA](https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html)


To generate kernel resource usage:

```bash
hipcc -c example.cpp -Rpass-analysis=kernel-resource-usage
```

To generate ISA source files (including `*.s`):

```bash
hipcc -c --save-temps -g example.cpp
```
