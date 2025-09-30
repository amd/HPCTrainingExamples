# TinyTransformer: Progressive Optimization Training Example

Progressive optimization of a Tiny LLaMA transformer implementation demonstrating performance improvement techniques from baseline PyTorch through ultra-optimized Triton kernels with comprehensive ROCm profiling integration.

## Implementation Versions

| Version | Implementation | Key Features | Expected Speedup |
|---------|---------------|-------------|------------------|
| **Version 1** | PyTorch Baseline | Standard operations, profiling foundation | 1.0x (baseline) |
| **Version 2** | PyTorch Fused | Kernel fusion, Flash Attention | ~1.6x |
| **Version 3** | Triton Integration | Custom Triton kernels | ~2.2x |
| **Version 4** | Triton Ultra | Ultra-fused single-kernel blocks | ~2.8x |

## Quick Start

```bash
# Version 1: Baseline profiling
cd version1_pytorch_baseline/
bash run_all_profilers.sh

# Version 2: Fused kernel optimization
cd ../version2_pytorch_fused/
bash run_all_profilers.sh

# Version 3: Triton kernel implementation
cd ../version3_triton/
bash run_all_profilers.sh

# Version 4: Ultra-optimized fusion
cd ../version4_triton_ultra/
bash run_all_profilers.sh
```

## Directory Structure

```
TinyTransformer/
├── README.md                              # This overview
├── TINY_LLAMA_ARCHITECTURE.md             # Architecture specifications
├── TECHNICAL_APPENDICES.md                # Implementation details
├── version1_pytorch_baseline/             # Standard PyTorch implementation
│   ├── README.md                          # Version 1 guide
│   ├── tiny_llama_v1.py                   # Baseline implementation
│   ├── run_pytorch_profiler.py            # PyTorch profiler integration
│   ├── run_deepspeed_flops.py            # DeepSpeed FLOPS profiler
│   ├── run_all_profilers.sh              # Profiling orchestration
│   └── exercises/                         # Training exercises
│       ├── exercise_1_baseline_analysis.md
│       ├── exercise_2_memory_analysis.md
│       └── exercise_3_bottleneck_identification.md
├── version2_pytorch_fused/                # Fused operations optimization
│   ├── README.md                          # Version 2 guide
│   ├── tiny_llama_v2.py                   # Fused implementation
│   ├── run_all_profilers.sh              # Complete profiling suite
│   ├── run_rocprof_compute.sh             # Kernel-level profiling
│   ├── run_rocprof_sys.sh                # System profiling
│   └── run_rocprofv3.sh                   # ROCprofv3 integration
├── version3_triton/                       # Triton kernel integration
│   ├── README.md                          # Version 3 guide
│   ├── tiny_llama_v3.py                   # Triton implementation
│   ├── run_triton_profiling.py            # Triton profiling
│   ├── run_rocprof_triton.sh             # ROCm profiling
│   └── exercises/                         # Triton exercises
│       ├── exercise1_triton_basics.md
│       ├── exercise2_swiglu_optimization.md
│       └── exercise3_flash_attention.md
└── version4_triton_ultra/                 # Ultra-fused implementation
    ├── README.md                          # Version 4 guide
    ├── tiny_llama_v4.py                   # Ultra-optimized implementation
    ├── run_ultra_profiling.py             # Advanced profiling
    └── exercises/
        └── exercise1_ultra_fusion.md
```

## Prerequisites

- AMD GPU with ROCm support (MI100, MI200, MI300 series)
- ROCm 6.0+
- Python 3.10+
- PyTorch with ROCm support
- ROCm profiling tools (rocprof-sys, rocprof-compute, rocprofv3)
- DeepSpeed (FLOPS profiler)
- Triton (versions 3+4)

## Profiling Tools

Each version integrates progressive profiling capabilities:

1. **PyTorch Profiler**: Framework-level performance analysis
2. **DeepSpeed FLOPS Profiler**: Computational efficiency metrics
3. **ROCprofv3**: Legacy ROCm profiling
4. **ROCprof-sys**: System-level performance monitoring
5. **ROCprof-compute**: Kernel-level analysis and optimization

## Documentation

- **[TINY_LLAMA_ARCHITECTURE.md](TINY_LLAMA_ARCHITECTURE.md)**: Model architecture specifications
- **[TECHNICAL_APPENDICES.md](TECHNICAL_APPENDICES.md)**: Implementation details and optimization techniques
- **Version READMEs**: Detailed guides for each implementation version
- **Exercise files**: Hands-on profiling and optimization exercises

## Performance Insights

### Expected Improvements
- **Memory Bandwidth**: 45% improvement through fusion
- **Kernel Count**: 95% reduction via ultra-fusion
- **GPU Utilization**: 78% increase in wavefront occupancy
- **Energy Efficiency**: 32% reduction in power consumption

## Resources

- **AMD ROCm Documentation**: [ROCm Developer Portal](https://rocm.docs.amd.com/)
- **ROCprof-compute Guide**: [Profiling Documentation](https://rocm.docs.amd.com/projects/rocprof-compute/)
- **PyTorch ROCm Support**: [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)