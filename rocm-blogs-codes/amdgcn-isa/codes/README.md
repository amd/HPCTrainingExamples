## Code examples for 'Reading AMDGCN ISA' blogpost

To generate kernel resource usage:

```bash
hipcc -c example.cpp -Rpass-analysis=kernel-resource-usage
```

To generate ISA source files (including `*.s`):
```bash
hipcc -c --save-temps -g example.cpp
```
