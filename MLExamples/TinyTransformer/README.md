

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

Performance figures for the small and medium configurations are summarized in the tables above and in [Key Performance Insights](#key-performance-insights).

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

### 0. Set up and verify environment
On the training cluster's compute node, load the modules (adjust names/versions for your site):

```bash
module load rocm pytorch
```

Then confirm ROCm, PyTorch, and the GPU(s) are setup correctly:

```bash
# Check ROCm installation
rocminfo

# Verify GPU is detected
rocm-smi

# Check PyTorch + ROCm
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 1. Run Version 1 (Baseline) - 5 minutes
```bash
cd version1_pytorch_baseline/
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 20

# Expected output:
# Loss: ~7.0
# Speed: ~373 samples/sec
# Memory: ~522 MB
```

For a deeper analysis with the PyTorch profiler, and visualizing the output in TensorBoard,
please follow the workshop exercises in
[version1_pytorch_baseline/README.md](https://github.com/amd/HPCTrainingExamples/tree/main/MLExamples/TinyTransformer/version1_pytorch_baseline#workshop-exercises).

### 2. Run Version 2 (Fused) - 5 minutes
```bash
cd version2_pytorch_fused
python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 30

# Expected output:
# Loss: 6.9310 
# Speed: 187.6 samples/sec (2x faster)
# Memory: 370.4 MB
```

To compare the baseline version1 to the fused version2 performance,
follow instructions in [version2_pytorch_fused/README.md](https://github.com/amd/HPCTrainingExamples/tree/main/MLExamples/TinyTransformer/version2_pytorch_fused#step-1-baseline-comparison).

Try profiling this workload with ROCm profilers using commands listed in
[version2_pytorch_fused/README.md](https://github.com/amd/HPCTrainingExamples/tree/main/MLExamples/TinyTransformer/version2_pytorch_fused#exercise-3-rocm-tools-deep-dive).
An example of using rocprofv3 on this example is provided below:

```bash
rocprofv3 --kernel-trace --stats --truncate-kernels -- python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 30
```
The above command produces a hotspot list of GPU kernels. The `--truncate-kernels` option helps remove arguments
from the kernel name for better readability.

### 3. Run Version 3 (Optimized) - 5 minutes
```bash
cd version3_triton/
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20

# Expected output:
# Loss: ~7.0 (same correctness!)
# Speed: ~2065 samples/sec (5.5x faster!)
# Memory: ~282 MB (46% less!)
```

An exercise similar to the one you did for version2 is recommended for
this version as well using ROCm profiling tools. As an example, you can
collect a comprehensive timeline trace with host and device activity
with `rocprof-sys` using the command below:

```bash
rocprof-sys-run --profile --trace -- python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 30
```
View the trace with [https://ui.perfetto.dev](https://ui.perfetto.dev).

### 4. Run Version 4 (Ultra optimized) - 5 minutes
```bash
cd version4_pytorch_sdpa/
python3 tiny_llama_v4.py
```

<!-- 
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
-->

## Directory Structure

Layout under `MLExamples/TinyTransformer/` in this repository:

```
TinyTransformer/
├── README.md
├── TINY_LLAMA_ARCHITECTURE.md
├── TECHNICAL_APPENDICES.md
├── version1_pytorch_baseline/
│   ├── tiny_llama_v1.py
│   ├── run_pytorch_profiler.py, run_deepspeed_flops.py, run_all_profilers.sh
│   ├── run_*.sh, launch_performance_study.sh
│   └── exercises/
│       ├── exercise_1_baseline_analysis.md
│       ├── exercise_2_memory_analysis.md
│       └── exercise_3_bottleneck_identification.md
├── version2_pytorch_fused/
│   ├── tiny_llama_v2.py
│   ├── run_*.py, run_*.sh, launch_performance_study.sh
│   └── exercises/
├── version3_triton/
│   ├── tiny_llama_v3.py, run_triton_profiling.py, run_rocprof_triton.sh
│   ├── launch_performance_study.sh
│   └── exercises/  (including performance_debugging/)
└── version4_pytorch_sdpa/
    ├── tiny_llama_v4.py, run_ultra_profiling.py, launch_performance_study.sh
    └── exercises/
        └── exercise1_ultra_fusion.md
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

MIT License — see the repository [`LICENSE.md`](../../LICENSE.md) at the git root of **HPCTrainingExamples**.

---

**Ready to start profiling?** Begin with [Quick Start](#quick-start) (environment modules and first runs) above.


