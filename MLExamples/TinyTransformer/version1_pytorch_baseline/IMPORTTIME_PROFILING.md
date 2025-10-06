# Python Import Time Profiling

## Overview

The `python -X importtime` flag provides detailed timing information about module imports during Python script execution. This is useful for identifying slow imports that can impact startup time and overall application performance.

## Basic Usage

```bash
python -X importtime script.py
```

This outputs a hierarchical tree showing:
- Import time for each module
- Cumulative time including sub-imports
- Self time (time spent in the module itself)

## Output Format

```
import time: self [us] | cumulative | imported package
import time:       150 |        150 |   _frozen_importlib_external
import time:        89 |         89 |     _codecs
import time:       658 |        747 |   codecs
import time:       597 |        597 |   encodings.aliases
import time:      1521 |       2865 | encodings
```

- **self [us]**: Time spent in the module itself (microseconds)
- **cumulative**: Total time including all sub-imports (microseconds)
- **imported package**: Module name with indentation showing import hierarchy

## Example: Profiling TinyLlama V1

### Basic Import Analysis

```bash
python -X importtime tiny_llama_v1.py 2> import_times.txt
```

This redirects the import timing output (stderr) to a file for analysis.

### Analyzing PyTorch Import Time

```bash
python -X importtime -c "import torch" 2>&1 | grep -E "torch|time:"
```

Expected output shows PyTorch's heavy import cost:
```
import time:   1234567 |   1234567 | torch
```

### Analyzing DeepSpeed Import Time

```bash
python -X importtime -c "import deepspeed" 2>&1 | grep -E "deepspeed|time:"
```

## Common Import Time Bottlenecks in AI Workloads

### 1. PyTorch (torch)
- Typical import time: 500ms - 2000ms
- Loads CUDA/ROCm libraries
- Initializes operator registry
- Sets up autograd engine

### 2. Transformers Library
- Typical import time: 300ms - 1000ms
- Loads tokenizers
- Registers model architectures
- Initializes configuration classes

### 3. DeepSpeed
- Typical import time: 200ms - 800ms
- Loads distributed training components
- Initializes optimization kernels
- Sets up communication backends

### 4. NumPy/SciPy
- Typical import time: 50ms - 200ms
- Loads optimized BLAS/LAPACK libraries
- Initializes array operations

## Best Practices

### 1. Lazy Imports
Move imports inside functions for code that's not always executed:

```python
def run_with_profiler():
    # Only import when profiler is actually used
    from torch.profiler import profile, ProfilerActivity
    ...
```

### 2. Conditional Imports
Import heavy dependencies only when needed:

```python
if args.enable_profiler:
    import deepspeed.profiling.flops_profiler as fp
```

### 3. Import Grouping
Organize imports by load time to understand startup cost:

```python
# Fast imports
import os
import sys
import argparse

# Medium imports
import numpy as np
import pandas as pd

# Heavy imports (consider lazy loading)
import torch
import deepspeed
```

## Optimization Techniques

### 1. Module-Level Import Caching
Python caches imports in `sys.modules`, so subsequent imports are fast:

```python
import torch  # Slow first time
import torch  # Fast - already cached
```

### 2. Using `__import__()` for Dynamic Imports
For plugins or optional features:

```python
def load_profiler(profiler_type):
    if profiler_type == "pytorch":
        torch_prof = __import__("torch.profiler", fromlist=["profile"])
        return torch_prof
```

### 3. Parallel Import Loading
Not natively supported, but can structure code to minimize import depth.

## Analyzing Import Time Results

### Generate Report
```bash
python -X importtime tiny_llama_v1.py 2>&1 | \
    grep "import time:" | \
    sort -k3 -n -r | \
    head -20 > top_imports.txt
```

### Parse with Script
```python
import re
import sys

with open('import_times.txt', 'r') as f:
    for line in f:
        match = re.search(r'import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.+)', line)
        if match:
            self_time = int(match.group(1))
            cumulative = int(match.group(2))
            module = match.group(3).strip()
            if cumulative > 100000:  # > 100ms
                print(f"{module}: {cumulative/1000:.2f}ms")
```

## ROCm/PyTorch Specific Considerations

### HIP Runtime Loading
ROCm's HIP runtime can add significant import overhead:
- libamdhip64.so loading
- GPU device detection
- Architecture-specific kernel initialization

### Environment Variables Impact
These can affect import time:
```bash
# Reduce logging overhead during import
AMD_LOG_LEVEL=0 MIOPEN_LOG_LEVEL=0 python -X importtime script.py

# Skip GPU initialization during import analysis
HIP_VISIBLE_DEVICES=-1 python -X importtime script.py
```

## Integration with Other Profiling Tools

### Combine with cProfile
```bash
# First check import time
python -X importtime script.py 2> imports.txt

# Then profile runtime
python -m cProfile -o profile.stats script.py
```

### Combine with PyTorch Profiler
```python
# Fast startup with lazy imports
def main():
    import torch
    from torch.profiler import profile

    # Your training code here
    ...

if __name__ == "__main__":
    main()
```

## Example Analysis for Version 1

### Expected Import Hierarchy

```
import time: self [us] | cumulative | imported package
import time:      2341 |       2341 |   _frozen_importlib_external
import time:    850000 |     850000 | torch               # Dominant cost
import time:    120000 |     120000 | torch.nn
import time:     45000 |      45000 | torch.optim
import time:     23000 |      23000 | apex.normalization.fused_layer_norm
import time:     18000 |      18000 | apex.transformer.functional.fused_rope
import time:      8000 |       8000 | argparse
import time:      3500 |       3500 | json
```

### Interpreting Results

- **torch**: Largest import cost (850ms typical)
- **torch.nn**: Additional overhead for neural network modules
- **apex**: NVIDIA optimizations (ROCm compatible)
- Standard library imports (argparse, json): Negligible cost

## When to Use Import Time Profiling

1. **Debugging slow script startup**: Identify which imports are causing delays
2. **Optimizing CLI tools**: Reduce time-to-first-output for user experience
3. **Container startup optimization**: Minimize cold-start latency
4. **CI/CD pipeline optimization**: Reduce test suite initialization time

## Limitations

- Does not profile runtime execution (use cProfile or PyTorch Profiler for that)
- Import time varies based on system load and cold vs. warm cache
- First import after system reboot will be slower due to OS page cache

## References

- [PEP 565 - Show DeprecationWarning in __main__](https://www.python.org/dev/peps/pep-0565/)
- [Python -X Options Documentation](https://docs.python.org/3/using/cmdline.html#id5)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
