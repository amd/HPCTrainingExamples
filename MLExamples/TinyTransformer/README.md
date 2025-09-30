# Castille AI Workshop: ROCm Tools for PyTorch AI Workload Profiling

## Workshop Overview

This hands-on workshop provides a comprehensive guide to profiling AI workloads using AMD ROCm tools and PyTorch. Through progressive optimization of a Tiny LLaMA transformer implementation, participants will master the complete profiling ecosystem from framework-level tools to hardware-specific profilers.

## Learning Objectives

By the end of this workshop, participants will be able to:
- Configure deterministic execution environments for reproducible profiling
- Use PyTorch native profiling tools for performance characterization
- Integrate DeepSpeed FLOPS profiler for computational intensity analysis
- Apply ROCm profiling tools (rocprofv3, rocprof-sys, rocprof-compute) for kernel-level optimization
- Implement progressive optimization techniques from kernel fusion to custom GPU programming
- Perform roofline analysis and bottleneck identification for production AI workloads

## Workshop Structure

This workshop follows a progressive optimization methodology with four implementation versions, each building upon the previous with enhanced profiling capabilities and performance improvements.

### Version Progression

| Version | Implementation | Key Features | Expected Speedup |
|---------|---------------|-------------|------------------|
| **Version 1** | PyTorch Baseline | Standard operations, profiling foundation | 1.0x (baseline) |
| **Version 2** | PyTorch Fused | Kernel fusion, Flash Attention | ~1.6x |
| **Version 3** | Triton Integration | Custom Triton kernels | ~2.2x |
| **Version 4** | Triton Ultra | Ultra-fused single-kernel blocks | ~2.8x |

### Profiling Tools Progression

Each version introduces additional profiling capabilities:

1. **PyTorch Profiler**: Framework-level performance analysis
2. **DeepSpeed FLOPS Profiler**: Computational efficiency metrics
3. **ROCprofv3**: Legacy ROCm profiling for compatibility
4. **ROCprof-sys**: System-level performance monitoring
5. **ROCprof-compute**: Advanced kernel-level analysis and optimization

## Prerequisites

### Hardware Requirements
- AMD GPU with ROCm support (MI100, MI200, MI300 series, or RX 6000/7000 series)
- Minimum 16GB system memory
- ROCm 6.0+ installed and configured

### Software Requirements
- Python 3.10+
- PyTorch with ROCm support
- ROCm profiling tools suite
- DeepSpeed (for FLOPS profiler)
- Triton (for advanced versions)

## Quick Start

### 1. Environment Setup
```bash
cd setup/
bash environment_setup.sh
python validation_scripts/test_environment.py
```

### 2. Run Version 1 (Baseline)
```bash
cd version1_pytorch_baseline/
python tiny_llama_v1.py --enable-pytorch-profiler --batch-size 8 --seq-len 128
```

### 3. Progressive Workshop Flow
```bash
# Version 1: Establish baseline performance
cd version1_pytorch_baseline/
bash run_all_profilers.sh

# Version 2: Kernel fusion optimization
cd ../version2_pytorch_fused/
bash run_all_profilers.sh

# Version 3: Triton kernel integration
cd ../version3_triton/
bash run_all_profilers.sh

# Version 4: Ultra-fused implementation
cd ../version4_triton_ultra/
bash run_all_profilers.sh

# Compare all versions
cd ../analysis_tools/
python compare_versions.py --generate-report
```

## Directory Structure

```
castille-ai-workshop-training/
├── README.md                              # This overview
├── setup/                                 # Environment and prerequisites
│   ├── environment_setup.md               # Detailed setup instructions
│   ├── environment_setup.sh               # Automated setup script
│   ├── requirements.txt                   # Python dependencies
│   └── validation_scripts/                # Environment validation
│       ├── test_environment.py            # Comprehensive environment test
│       ├── test_rocm_installation.py      # ROCm stack validation
│       └── test_profiling_tools.py        # Profiling tools validation
├── version1_pytorch_baseline/             # Standard PyTorch implementation
│   ├── README.md                          # Detailed guided instructions
│   ├── tiny_llama_v1.py                   # Enhanced baseline implementation
│   ├── run_pytorch_profiler.py            # PyTorch profiler integration
│   ├── run_deepspeed_flops.py            # DeepSpeed FLOPS profiler
│   ├── run_all_profilers.sh              # Orchestrated profiling script
│   └── exercises/                         # Hands-on exercises and analysis
│       ├── exercise_1_baseline_analysis.md
│       ├── exercise_2_memory_analysis.md
│       └── exercise_3_bottleneck_identification.md
├── version2_pytorch_fused/                # Fused operations optimization
│   ├── README.md                          # Fusion optimization guide
│   ├── tiny_llama_v2.py                   # Fused implementation
│   ├── run_pytorch_profiler.py            # Enhanced PyTorch profiling
│   ├── run_deepspeed_flops.py            # FLOPS analysis
│   ├── run_rocprofv3.sh                   # ROCprofv3 integration
│   ├── run_rocprof_sys.sh                # System profiling
│   ├── run_rocprof_compute.sh             # Kernel-level profiling
│   ├── run_all_profilers.sh              # Complete profiling suite
│   └── exercises/                         # Advanced profiling exercises
│       ├── exercise_1_fusion_analysis.md
│       ├── exercise_2_flash_attention.md
│       └── exercise_3_rocm_tools_intro.md
├── version3_triton/                       # Triton kernel integration
│   ├── README.md                          # Triton optimization guide
│   ├── tiny_llama_v3.py                   # Triton-enhanced implementation
│   ├── triton_kernels.py                  # Custom Triton kernels
│   ├── run_pytorch_profiler.py            # Framework profiling
│   ├── run_deepspeed_flops.py            # Computational analysis
│   ├── run_rocprofv3.sh                   # Legacy profiling
│   ├── run_rocprof_sys.sh                # System monitoring
│   ├── run_rocprof_compute.sh             # Advanced kernel analysis
│   ├── run_all_profilers.sh              # Complete profiling
│   └── exercises/                         # Triton development exercises
│       ├── exercise_1_triton_basics.md
│       ├── exercise_2_custom_kernels.md
│       └── exercise_3_performance_tuning.md
├── version4_triton_ultra/                 # Ultra-fused implementation
│   ├── README.md                          # Ultra-optimization guide
│   ├── tiny_llama_v4.py                   # Ultra-fused implementation
│   ├── triton_ultra_kernels.py            # Ultra-fused kernels
│   ├── [profiling scripts]                # Complete profiling suite
│   └── exercises/                         # Advanced optimization
│       ├── exercise_1_ultra_fusion.md
│       ├── exercise_2_register_optimization.md
│       └── exercise_3_production_deployment.md
├── analysis_tools/                        # Performance analysis utilities
│   ├── compare_versions.py                # Cross-version performance comparison
│   ├── roofline_analysis.py               # Roofline model implementation
│   ├── performance_dashboard.py           # Interactive performance dashboard
│   ├── regression_tester.py               # Automated regression testing
│   └── report_generator.py                # Comprehensive report generation
└── slides/                                # Presentation materials
    └── luka_presentation_materials/        # Castille AI workshop slides
        ├── workshop_overview.pptx
        ├── profiling_methodology.pptx
        ├── optimization_techniques.pptx
        └── results_analysis.pptx
```

## Workshop Execution Timeline

### Session 1: Foundation (45 minutes)
- Environment setup and validation
- Version 1 baseline profiling
- PyTorch profiler introduction
- Performance characterization methodology

### Session 2: Optimization (60 minutes)
- Version 2 kernel fusion techniques
- ROCm tools introduction
- Memory optimization analysis
- Comparative performance analysis

### Session 3: Advanced Techniques (60 minutes)
- Version 3 Triton kernel development
- Custom GPU programming
- Advanced profiling techniques
- Production optimization strategies

### Session 4: Mastery (45 minutes)
- Version 4 ultra-fusion implementation
- Complete profiling suite utilization
- Roofline analysis and bottleneck resolution
- Workshop wrap-up and next steps

## Key Performance Insights

### Expected Performance Progression
- **Memory Bandwidth**: 45% improvement through fusion
- **Kernel Count**: 95% reduction via ultra-fusion
- **GPU Utilization**: 78% increase in wavefront occupancy
- **Energy Efficiency**: 32% reduction in power consumption

### Profiling Tool Capabilities
- **PyTorch Profiler**: Framework overhead, operator timing
- **DeepSpeed FLOPS**: Computational intensity, arithmetic utilization
- **ROCprof-compute**: Kernel execution patterns, memory hierarchy analysis
- **ROCprof-sys**: System-wide performance monitoring
- **Roofline Analysis**: Theoretical performance bounds, optimization guidance

## Contributing

This workshop is designed for continuous improvement. Contributions are welcome:
- Additional optimization techniques
- Enhanced profiling methodologies
- Extended GPU architecture support
- Advanced analysis tools

## Support and Resources

- **Workshop Issues**: Submit GitHub issues for technical problems
- **AMD ROCm Documentation**: [ROCm Developer Portal](https://rocm.docs.amd.com/)
- **ROCprof-compute Guide**: [Profiling Documentation](https://rocm.docs.amd.com/projects/rocprof-compute/)
- **PyTorch ROCm Support**: [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)

## Authors and Acknowledgments

Developed for the Castille AI Workshop (October 16, 2024) by HPC/AI performance engineers with extensive experience optimizing production ML workloads on AMD GPU infrastructure.

## License

MIT License - See LICENSE file for details

---

**Ready to start profiling? Begin with the [Environment Setup Guide](setup/environment_setup.md)**