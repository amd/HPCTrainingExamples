

# AI Workshop: ROCm Tools for PyTorch AI Workload Profiling

README.md from `HPCTrainingExamples/MLExamples/TinyTransformer` in the Training Examples repository

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

### Small Configuration (Quick Start)
**Config:** Hidden=512, Layers=8, SeqLen=128, Batch=8

| Version | Speed (samples/sec) | Batch Time (ms) | Forward (ms) | Backward (ms) | Memory (MB) | Speedup |
|---------|---------------------|-----------------|--------------|---------------|-------------|---------|
| **V1 Baseline** | 372.9 | 21.7 | 10.8 | 9.2 | 522.3 | 1.0x |
| **V3 Triton** | 2,065.0 | 3.9 | 3.2 | 0.3 | 281.8 | **5.5x** |

### Medium Configuration (Production Scale)
**Config:** Hidden=1024, Layers=12, SeqLen=512, Batch=16

| Version | Throughput (tok/s) | Batch (ms) | Forward (ms) | Backward (ms) | Optimizer (ms) | Memory (MB) | Speedup |
|---------|-------------------|------------|--------------|---------------|----------------|-------------|---------|
| **V1 Baseline** | 50,017 | 163.8 | 50.3 | 107.4 | 6.1 | 2,358.7 | 1.0x |
| **V2 Fused** | 60,192 | 136.1 | 44.8 | 85.6 | 5.8 | 2,358.9 | 1.20x |
| **V3 Triton** | 156,652 | 52.3 | 51.3 | 0.6 | 0.4 | 916.2 | **3.13x** |
| **V4 Ultra** | 157,169 | 52.1 | 51.1 | 0.6 | 0.4 | 916.5 | **3.14x** |

**See [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md) for complete analysis**

### Profiling Tools Progression

Each version introduces additional profiling capabilities:

1. **PyTorch Profiler**: Framework-level performance analysis
2. **DeepSpeed FLOPS Profiler**: Computational efficiency metrics
3. **rocprofv3**: GPU hotspots, device activity tracing and hardware counter collection
4. **rocprof-sys**: System-level performance monitoring
5. **rocprof-compute**: Advanced kernel-level analysis and optimization

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

### 0. Set up environment
On the training cluster's compute node, the required environment may be set up using the following
commands:

```bash
module load rocm pytorch openmpi rocprofiler-compute rocprofiler-systems/develop
```

### 1. Verify Environment
```bash
# Check ROCm installation
rocminfo

# Verify GPU is detected
rocm-smi

# Check PyTorch + ROCm
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 2. Run Version 1 (Baseline) - 5 minutes
```bash
cd version1_pytorch_baseline/
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 20

# Expected output:
# Loss: ~7.0
# Speed: ~373 samples/sec
# Memory: ~522 MB
```

### 3. Run Version 3 (Optimized) - 5 minutes
```bash
cd version3_triton/
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20

# Expected output:
# Loss: ~7.0 (same correctness!)
# Speed: ~2065 samples/sec (5.5x faster!)
# Memory: ~282 MB (46% less!)
```

### 4. Performance Debugging Exercise - 15 minutes
```bash
cd version3_triton/exercises/performance_debugging/

# Run all optimization stages with profiling
./run_all_stages.sh

# This demonstrates the complete debugging journey:
# Stage 1: Broken (loss=942) → Fix weight init
# Stage 2: Slow (15 samp/s) → Fix tensor layout
# Stage 3: Fast (2065 samp/s) → Optimal!
```

### 5. Profile with ROCm Tools (Optional)
```bash
# Basic profiling
rocprofv3 --stats --kernel-trace --truncate-kernels -- python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10
```
The above command produces a hotspot list of GPU kernels. The `--truncate-kernels` option helps remove arguments
from the kernel name for better readability.

```bash
# Detailed kernel trace
rocprofv3 --runtime-trace --output-format pftrace -- python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 5
```
The above command generates a Perfetto trace file with a timeline view of GPU kernels, memory copies
to and from device, runtime API activity, and any ROCtx markers. View the trace at
[https://ui.perfetto.dev](https://ui.perfetto.dev).

## Directory Structure

```
castille-ai-workshop-training/
 README.md                              # This overview
 setup/                                 # Environment and prerequisites
    environment_setup.md               # Detailed setup instructions
    environment_setup.sh               # Automated setup script
    requirements.txt                   # Python dependencies
    validation_scripts/                # Environment validation
        test_environment.py            # Comprehensive environment test
        test_rocm_installation.py      # ROCm stack validation
        test_profiling_tools.py        # Profiling tools validation
 version1_pytorch_baseline/             # Standard PyTorch implementation
    README.md                          # Detailed guided instructions
    tiny_llama_v1.py                   # Enhanced baseline implementation
    run_pytorch_profiler.py            # PyTorch profiler integration
    run_deepspeed_flops.py            # DeepSpeed FLOPS profiler
    run_all_profilers.sh              # Orchestrated profiling script
    exercises/                         # Hands-on exercises and analysis
        exercise_1_baseline_analysis.md
        exercise_2_memory_analysis.md
        exercise_3_bottleneck_identification.md
 version2_pytorch_fused/                # Fused operations optimization
    README.md                          # Fusion optimization guide
    tiny_llama_v2.py                   # Fused implementation
    run_pytorch_profiler.py            # Enhanced PyTorch profiling
    run_deepspeed_flops.py            # FLOPS analysis
    run_rocprofv3.sh                   # rocprofv3 integration
    run_rocprof_sys.sh                # System profiling
    run_rocprof_compute.sh             # Kernel-level profiling
    run_all_profilers.sh              # Complete profiling suite
    exercises/                         # Advanced profiling exercises
        exercise_1_fusion_analysis.md
        exercise_2_flash_attention.md
        exercise_3_rocm_tools_intro.md
 version3_triton/                       # Triton kernel integration
    README.md                          # Triton optimization guide
    tiny_llama_v3.py                   # Triton-enhanced implementation
    triton_kernels.py                  # Custom Triton kernels
    run_pytorch_profiler.py            # Framework profiling
    run_deepspeed_flops.py            # Computational analysis
    run_rocprofv3.sh                   # Legacy profiling
    run_rocprof_sys.sh                # System monitoring
    run_rocprof_compute.sh             # Advanced kernel analysis
    run_all_profilers.sh              # Complete profiling
    exercises/                         # Triton development exercises
        exercise_1_triton_basics.md
        exercise_2_custom_kernels.md
        exercise_3_performance_tuning.md
 version4_pytorch_sdpa/                 # Ultra-fused implementation
    README.md                          # Ultra-optimization guide
    tiny_llama_v4.py                   # Ultra-fused implementation
    triton_ultra_kernels.py            # Ultra-fused kernels
    [profiling scripts]                # Complete profiling suite
    exercises/                         # Advanced optimization
        exercise_1_ultra_fusion.md
        exercise_2_register_optimization.md
        exercise_3_production_deployment.md
 analysis_tools/                        # Performance analysis utilities
    compare_versions.py                # Cross-version performance comparison
    roofline_analysis.py               # Roofline model implementation
    performance_dashboard.py           # Interactive performance dashboard
    regression_tester.py               # Automated regression testing
    report_generator.py                # Comprehensive report generation
 slides/                                # Presentation materials
     luka_presentation_materials/        # AI workshop slides
         workshop_overview.pptx
         profiling_methodology.pptx
         optimization_techniques.pptx
         results_analysis.pptx
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

### Actual Performance Results (AMD MI325X, ROCm 6.4.4, PyTorch 2.7.1)

**Test Configuration:** Batch=8, SeqLen=128, Hidden=512, Layers=8, Heads=8

| Metric | V1 Baseline | V3 Optimized | Improvement |
|--------|-------------|--------------|-------------|
| **Training Speed** | 372.9 samples/sec | 2065.0 samples/sec | **5.5x faster** |
| **Batch Time** | 21.7 ms | 3.9 ms | **5.6x faster** |
| **Forward Pass** | 10.8 ms | 3.2 ms | **3.4x faster** |
| **Memory Usage** | 522.3 MB | 281.8 MB | **46% reduction** |
| **Throughput** | 47,735 tokens/sec | 264,320 tokens/sec | **5.5x faster** |

### Key Optimization Techniques Applied

1. **Flash Attention** (Memory-Efficient Attention)
   - **V3**: Custom Triton Flash Attention kernel
   - **V4**: PyTorch SDPA (hardware-accelerated)
   - Both achieve ~3.1x speedup through memory-efficient attention
   - Result: 46% memory reduction, 61% less memory bandwidth

2. **Tensor Contiguity** (`.contiguous()` after GQA operations)
   - Ensures optimal memory layout for Triton kernels
   - Fixes stride-related performance issues
   - Result: 20x speedup over non-contiguous version

3. **Hybrid Kernel Strategy**
   - Use Triton for: RMSNorm, Flash Attention (memory-bound ops)
   - Use PyTorch/rocBLAS for: Matrix multiplies (compute-bound ops)
   - Don't write custom Triton kernels for matmuls - rocBLAS is already optimal
   - Result: 3.1x overall speedup

4. **Proper Weight Initialization** (`std=0.02`)
   - Critical for correct logits scale
   - Prevents exploding/vanishing gradients
   - Result: Loss goes from 942 → 7.0

### V3 vs V4: Two Paths to the Same Performance

- **V3 (Triton Custom Kernels)**: Custom Triton RMSNorm + Triton Flash Attention
- **V4 (PyTorch Optimized)**: PyTorch ops + PyTorch SDPA
- **Both achieve 3.1x speedup** - demonstrates that highly-optimized PyTorch operations can match custom kernels

### Profiling Tool Capabilities

- **PyTorch Profiler**: Framework overhead, operator timing, memory tracking
- **rocprofv3**: Kernel execution stats, device activity and runtime API timeline tracing, hardware counter collection
- **Manual Timing**: CUDA synchronization for accurate GPU timing

## Contributing

This workshop is designed for continuous improvement. Contributions are welcome:

- Additional optimization techniques
- Enhanced profiling methodologies
- Extended GPU architecture support
- Advanced analysis tools

## Support and Resources

- **Workshop Issues**: Submit GitHub issues for technical problems
- **AMD ROCm Documentation**: [ROCm Developer Portal](https://rocm.docs.amd.com/)
- **rocprofv3 tool usage**: [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#using-rocprofv3)
- **rocprof-sys Guide**: [rocprof-sys documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/index.html#rocm-systems-profiler-documentation)
- **rocprof-compute Guide**: [rocprof-compute Documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/#rocm-compute-profiler-documentation)
- **PyTorch ROCm Support**: [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)

## Authors and Acknowledgments

Developed for the CASTIEL AI Workshop (October 16, 2024) by HPC/AI performance engineers with extensive experience optimizing production ML workloads on AMD GPU infrastructure.

## License

MIT License - See LICENSE file for details

---

**Ready to start profiling? Begin with the [Environment Setup Guide](setup/environment_setup.md)**


