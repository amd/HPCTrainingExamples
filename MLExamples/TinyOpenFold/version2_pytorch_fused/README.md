# TinyOpenFold V2: PyTorch Fused - Kernel Fusion and ROCm Tools Integration

Educational implementation of AlphaFold 2's Evoformer architecture with comprehensive kernel fusion optimizations and ROCm profiling integration.

## Overview

Version 2 demonstrates the power of kernel fusion and introduces comprehensive ROCm profiling tools. Building on the baseline analysis from Version 1, this version implements targeted optimizations to achieve significant performance improvements through strategic kernel fusion, Flash Attention, and advanced ROCm profiling integration.

## Learning Objectives

After completing this version, you will be able to:

- Implement QKV fusion for MSA and triangle attention operations
- Integrate Flash Attention for memory-efficient attention computation
- Apply gate/proj fusion in triangle multiplicative updates
- Use ROCm profiling tools (rocprofv3, rocprof-sys-python, rocprof-compute) for hardware-level analysis
- Analyze kernel fusion impact on performance and memory usage
- Interpret ROCm profiling data for optimization insights
- Conduct ablation studies to quantify fusion benefits

## Key Optimizations Implemented

### 1. MSA QKV Fusion

- **Problem**: Separate Q, K, V linear projections create 3 kernel launches per attention operation
- **Solution**: Fused QKV projection with single kernel launch for both row and column attention
- **Expected Benefit**: 20-30% reduction in MSA attention overhead

### 2. Triangle QKV Fusion

- **Problem**: Separate Q, K, V projections in triangle attention (starting and ending)
- **Solution**: Combined QKV projections for both triangle attention variants
- **Expected Benefit**: 20-30% reduction in triangle attention overhead

### 3. Flash Attention Integration

- **Problem**: Standard attention has O(n²) memory complexity
- **Solution**: PyTorch's scaled_dot_product_attention with Flash Attention
- **Expected Benefit**: 50-80% memory reduction, enables larger sequences

### 4. Triangle Gate/Proj Fusion

- **Problem**: Separate gate and proj projections in triangle multiplicative updates
- **Solution**: Combined gate/proj computation with element-wise operations
- **Expected Benefit**: 15-25% triangle operation speedup

### 5. Torch Compile Integration

- **Problem**: Remaining kernel launch overhead
- **Solution**: Automatic fusion through torch.compile()
- **Expected Benefit**: Additional 10-20% speedup through automatic optimizations

## Quick Start

### Environment Setup

Before running V2, ensure your environment is set up correctly. See the [Environment Setup and Installation](../README.md#environment-setup-and-installation) section in the main README for detailed instructions.

**Quick summary:**
- Load modules: `module load python/3.12 rocm/7.2` (or `cray-python rocm/7.2`)
- Create and activate venv: `python3 -m venv venv && source venv/bin/activate`
- Install PyTorch (ROCm 7.1 nightly): `pip3 install torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/nightly/rocm7.1`
- Install DeepSpeed: `pip3 install deepspeed`
- Set up `LD_LIBRARY_PATH` for library loading

See the main README for complete setup instructions.

### Basic Fused Training

```bash
# Ensure you're in the version2_pytorch_fused directory
cd version2_pytorch_fused

# Default configuration with all fusions enabled
python3 tiny_openfold_v2.py --batch-size 4 --seq-len 64

# Expected output shows fusion statistics:
#   MSA QKV Fusion: Enabled
#   Triangle QKV Fusion: Enabled
#   Flash Attention: Enabled
#   Triangle Gate/Proj Fusion: Enabled
#   Kernel Reduction: 80.0% (48 fewer kernels)
```

### Validation Check

```bash
# Verify fusion optimizations work correctly
python3 tiny_openfold_v2.py --validate-setup

# Should output:
# V2 validation successful! Fusion setup working properly.
```

### Compare Fusion vs Baseline

```bash
# Compare all fusion enabled vs baseline (all fusion disabled)
python3 tiny_openfold_v2.py --compare-fusion --batch-size 4 --num-steps 50

# Output shows:
# - Training speed comparison (speedup)
# - Memory usage comparison (reduction)
# - Batch time comparison (improvement)
# - Kernel reduction percentage
```

### Enable All Fusions

```bash
# Explicitly enable all fusion optimizations
python3 tiny_openfold_v2.py --enable-all-fusion --batch-size 4
```

### Baseline Comparison Mode

```bash
# Run with all fusions disabled (equivalent to V1)
python3 tiny_openfold_v2.py --disable-all-fusion --batch-size 4
```

## Architecture Enhancements and Fusion Techniques

### Mathematical Foundation of Kernel Fusion

Kernel fusion combines multiple operations into a single GPU kernel to reduce memory bandwidth requirements and kernel launch overhead.

#### Fusion Efficiency Analysis

**Memory Bandwidth Reduction:**

For QKV Fusion:
- **Separate operations**: 3 × (Input Read + Weight Read + Output Write)
- **Fused operation**: Input Read + 3 × Weight Read + Output Write
- **Reduction**: ~40% for typical batch sizes (eliminates 2 redundant input reads)

**Kernel Launch Overhead:**
- Each kernel launch: 5-50 μs depending on operation size
- QKV fusion: 3 launches → 1 launch (saves 10-100 μs per attention)
- Triangle fusion: 4 launches → 2 launches (saves 10-100 μs per triangle op)

### 1. MSA QKV Fusion Implementation

#### Before Fusion (Baseline)

```python
# Three separate linear projections - 3 kernel launches
q = self.q_proj(msa)  # Kernel 1: GEMM [B,N,S,D] × [D,D] = [B,N,S,D]
k = self.k_proj(msa)  # Kernel 2: GEMM [B,N,S,D] × [D,D] = [B,N,S,D]
v = self.v_proj(msa)  # Kernel 3: GEMM [B,N,S,D] × [D,D] = [B,N,S,D]

# Memory reads: 3x MSA tensor + 3x weight matrices
# Memory writes: 3x output tensors
```

#### After Fusion (Optimized)

```python
# Single fused projection - 1 kernel launch
qkv = self.qkv_proj(msa)  # Kernel 1: GEMM [B,N,S,D] × [D,3D] = [B,N,S,3D]
q, k, v = qkv.chunk(3, dim=-1)  # Tensor view operation (no memory copy)

# Memory reads: 1x MSA tensor + 1x weight matrix (3x size)
# Memory writes: 1x output tensor (3x size)
# Bandwidth reduction: ~40% (eliminated 2 redundant MSA reads)
```

#### Implementation Details

```python
class FusedMSARowAttention(nn.Module):
    def __init__(self, config, fusion_config):
        super().__init__()
        if fusion_config.enable_qkv_fusion_msa:
            # Fused QKV projection - 3 operations combined into 1
            self.qkv_proj = nn.Linear(config.msa_dim, 3 * config.msa_dim, bias=False)
        else:
            # Separate projections (baseline)
            self.q_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
            self.k_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
            self.v_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
```

### 2. Flash Attention Deep Dive

#### Memory Complexity Analysis

**Standard Attention Memory:**
- Attention Matrix: O(B × H × S²)
- For S=64: 64² = 4,096 elements per head
- Total Memory: B × H × S² × 4 bytes
- Example: 4 × 4 × 64² × 4 = 262 KB per MSA sequence

**Flash Attention Memory:**
- Block Size: Typically 64 × 64
- Memory Usage: O(B × H × S) (linear in sequence length!)
- Reduction: S-fold memory reduction (64x for S=64)

#### Flash Attention Benefits

```python
# Use PyTorch's optimized Flash Attention
if self.fusion_config.enable_flash_attention:
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=pair_bias,  # Supports attention bias
        dropout_p=0.0,
        is_causal=False
    )
```

**Performance Characteristics:**
- Memory: O(S) instead of O(S²)
- Speed: 2-4x faster for sequences > 32
- Numerical stability: Built-in overflow protection

### 3. Triangle Fusion Implementation

#### Triangle Multiplicative Update Fusion

**Before Fusion:**
```python
# Four separate projections - 4 kernel launches
left = self.left_proj(pair)    # Kernel 1
right = self.right_proj(pair)  # Kernel 2
left_g = self.left_gate(pair)  # Kernel 3
right_g = self.right_gate(pair)  # Kernel 4

left = left * torch.sigmoid(left_g)
right = right * torch.sigmoid(right_g)
```

**After Fusion:**
```python
# Two fused projections - 2 kernel launches
proj = self.left_right_proj(pair)  # Kernel 1: Combined left+right
left, right = proj.chunk(2, dim=-1)

gate = self.left_right_gate(pair)  # Kernel 2: Combined gates
left_g, right_g = gate.chunk(2, dim=-1)

left = left * torch.sigmoid(left_g)
right = right * torch.sigmoid(right_g)
# Reduction: 4 kernels → 2 kernels (50% fewer launches)
```

### 4. Torch Compile Integration

```python
# Apply torch.compile for automatic fusion
if fusion_config.enable_torch_compile:
    model = torch.compile(
        model,
        mode='default',  # or 'max-autotune' for aggressive optimization
        dynamic=False
    )
```

**Torch Compile Optimizations:**
- Automatic elementwise operation fusion
- Memory layout optimization
- Shape specialization
- AMD GPU-specific optimizations

## Fusion Performance Analysis Framework

### Kernel Count Analysis

**Per Evoformer Block:**
- **Baseline**: 15 major kernel launches
  - MSA row attention: 3 (Q,K,V)
  - MSA column attention: 3 (Q,K,V)
  - Triangle mult out: 4 (left_proj, right_proj, left_gate, right_gate)
  - Triangle mult in: 4 (left_proj, right_proj, left_gate, right_gate)
  - Triangle attn start: 3 (Q,K,V)
  - Triangle attn end: 3 (Q,K,V)
  - Other ops: ~5 (transitions, outer product, etc.)

- **With All Fusions**: 3 major kernels
  - MSA row attention: 1 (fused QKV)
  - MSA column attention: 1 (fused QKV)
  - Triangle mult out: 2 (fused proj, fused gate)
  - Triangle mult in: 2 (fused proj, fused gate)
  - Triangle attn start: 1 (fused QKV)
  - Triangle attn end: 1 (fused QKV)
  - Other ops: ~5 (unchanged)

- **Kernel Reduction**: 12 kernels per block (80% reduction in attention/triangle ops)

### Expected Performance Gains

| Optimization | Impact | Memory Reduction | Kernel Reduction | Implementation Effort |
|-------------|--------|------------------|------------------|---------------------|
| **MSA QKV Fusion** | 1.2-1.4x | 15-25% | 67% (6→2 kernels) | Low |
| **Triangle QKV Fusion** | 1.2-1.3x | 15-25% | 67% (6→2 kernels) | Low |
| **Flash Attention** | 1.3-2.0x | 50-80% | Attention optimized | Medium |
| **Triangle Fusion** | 1.1-1.3x | 10-20% | 50% (8→4 kernels) | Low |
| **Torch Compile** | 1.1-1.2x | 5-10% | 10-30% | Very Low |
| **Combined Effect** | **1.5-2.2x** | **50-80%** | **60-80%** | - |

## Profiling and Analysis

### PyTorch Profiler with Fusion Analysis

```bash
# Basic profiling with fusion analysis
python3 run_pytorch_profiler.py --batch-size 4 --profile-dir ./fusion_analysis

# View comprehensive report
less fusion_analysis/comprehensive_profiling_report.md

# Compare with baseline (all fusions disabled)
python3 run_pytorch_profiler.py --disable-all-fusion --profile-dir ./baseline_analysis
```

**Provides:**
- Fusion-specific kernel analysis
- Kernel count reduction measurement
- Flash Attention performance tracking
- Memory bandwidth utilization

### ROCm Profiling Suite

AMD offers three performance profiling tools for ROCm-based applications:

#### 1. rocprofv3 - Kernel Statistics

```bash
# Basic kernel profiling
./run_rocprofv3.sh --batch-size 4 --seq-len 64

# View kernel statistics
less rocprofv3_profiles_v2/rocprofv3_summary.txt
```

**Key Metrics:**
- Kernel execution times
- Kernel call counts (verify fusion effectiveness)
- GPU utilization

#### 2. rocprof-sys-python - Python Call Stack Profiling

`rocprof-sys-python` provides Python call stack profiling with source-level instrumentation, enabling detailed analysis of function call counts and timing.

```bash
# Basic profiling with defaults (batch-size=2, seq-len=16 for smaller output)
./run_rocprof_sys.sh

# Custom batch size and sequence length
./run_rocprof_sys.sh --batch-size 4 --seq-len 64

# Direct command-line usage
rocprof-sys-python --trace -- ./tiny_openfold_v2.py --batch-size 2 --seq-len 16
```

**Output Files:**
- **ROCPD format** (`.rocpd` or `.rocpd.json`) - Recommended for AI/ML workloads with better thread support
- **Perfetto trace** (`.proto`) - Timeline visualization
- **Call stack data** (`trip_count-*.txt/json`, `wall_clock-*.txt/json`) - Function call counts and timing
- **Metadata** (`metadata-*.json`, `functions-*.json`) - Function and source information

**Visualization:**
```bash
# For Perfetto traces:
# 1. Copy .proto file to your local machine
# 2. Open https://ui.perfetto.dev in your browser
# 3. Click 'Open trace file' and select the .proto file

# For ROCPD format:
# Use ROCm tools or compatible viewers for AI/ML workload analysis
```

**Key Insights:**
- Python function call stack with call counts
- Function-level timing (wall clock, CPU time)
- CPU-GPU synchronization patterns
- Memory usage tracking (peak RSS, page RSS)
- Thread-level profiling

**Documentation:**
- ROCm Systems Profiler Python Guide: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html

**Note:** Default batch size (2) and sequence length (16) are optimized for profiling to reduce output file sizes. For production analysis, use larger values with `--batch-size` and `--seq-len` flags.

#### 3. rocprof-compute - Hardware Analysis

```bash
# Generate roofline plots
./run_rocprof_compute.sh --roof-only --batch-size 4

# Full profile with dispatch analysis
./run_rocprof_compute.sh --batch-size 4

# Analyze specific dispatch
./run_rocprof_compute.sh --mode analyze --dispatch 1538
```

**Key Metrics:**
- Roofline analysis (compute vs memory bound)
- Memory bandwidth utilization
- Hardware counter analysis

### Comprehensive Profiling Suite

```bash
# Run all profilers in one go
./run_all_profilers.sh --batch-size 4 --seq-len 64

# Quick profiling (skip rocprof-sys)
./run_all_profilers.sh --quick --batch-size 4

# View summary
less complete_profiling_*/PROFILING_SUMMARY.md
```

## Ablation Studies

### Testing Individual Fusions

```bash
# Only MSA QKV fusion
python3 tiny_openfold_v2.py \
    --disable-qkv-fusion-triangle \
    --disable-flash-attention \
    --disable-triangle-fusion

# Only Flash Attention
python3 tiny_openfold_v2.py \
    --disable-qkv-fusion-msa \
    --disable-qkv-fusion-triangle \
    --disable-triangle-fusion

# Only Triangle fusion
python3 tiny_openfold_v2.py \
    --disable-qkv-fusion-msa \
    --disable-qkv-fusion-triangle \
    --disable-flash-attention
```

### Automated Ablation Study

```bash
# Run comprehensive ablation study
./run_pytorch_profiler.sh --ablation --batch-size 4

# Results saved to pytorch_profiles_v2_ablation_*/
```

## Performance Study Launcher

```bash
# Standard performance study across configurations
./launch_performance_study.sh \
    --batch-sizes "2 4 8" \
    --seq-lens "32 64 128" \
    --num-runs 3

# Include baseline comparison
./launch_performance_study.sh --num-runs 3

# Include ablation study
./launch_performance_study.sh --ablation --num-runs 3

# View results
cat performance_study_*/results_summary.json
```

## Comparison with Version 1

### Running Comparative Analysis

```bash
# Run V1 baseline
cd ../version1_pytorch_baseline
python3 tiny_openfold_v1.py --batch-size 4 --seq-len 64 --num-steps 50 \
    --profile-dir ./v1_comparison

# Run V2 with comparison
cd ../version2_pytorch_fused
python3 tiny_openfold_v2.py --batch-size 4 --seq-len 64 --num-steps 50 \
    --compare-with-v1 ../version1_pytorch_baseline/v1_comparison/performance_summary.json
```

### Expected Improvements

Based on the fusion optimizations:
- **Speedup**: 1.5-2.2x training throughput
- **Memory**: 50-80% reduction (with Flash Attention)
- **Kernel Count**: 60-80% reduction in attention/triangle kernels
- **GPU Utilization**: Improved from better kernel efficiency

## Command Reference

### Model Configuration

```bash
--msa-dim 64              # MSA representation dimension
--pair-dim 128            # Pair representation dimension  
--num-blocks 4            # Number of Evoformer blocks
--num-seqs 16             # Number of MSA sequences
--seq-len 64              # Sequence length (residues)
```

### Training Parameters

```bash
--num-steps 50            # Training iterations
--batch-size 4            # Batch size
--learning-rate 3e-4      # Learning rate
--use-amp                 # Enable mixed precision (FP16)
```

### Fusion Configuration

```bash
# Enable/disable specific fusions
--enable-qkv-fusion-msa          # MSA QKV fusion (default: on)
--disable-qkv-fusion-msa         # Disable MSA QKV fusion
--enable-qkv-fusion-triangle     # Triangle QKV fusion (default: on)
--disable-qkv-fusion-triangle    # Disable triangle QKV fusion
--enable-flash-attention         # Flash Attention (default: on)
--disable-flash-attention        # Disable Flash Attention
--enable-triangle-fusion         # Triangle gate/proj fusion (default: on)
--disable-triangle-fusion        # Disable triangle fusion
--enable-torch-compile           # Enable torch.compile
--torch-compile-mode default     # Torch compile mode

# Fusion presets
--enable-all-fusion              # Enable everything
--disable-all-fusion             # Baseline mode (no fusions)
```

### Profiling Options

```bash
--enable-pytorch-profiler # Enable PyTorch profiler
--enable-memory-profiling # Track memory usage
--enable-rocm-profiling   # Enable ROCm tools integration
--enable-all-profiling    # Enable all profiling
--profile-dir PATH        # Output directory
```

## Code Structure

### Main Fusion Classes

**`FusionConfig`**: Configuration dataclass for fusion options

**`FusedMSARowAttention`**: MSA row attention with QKV fusion + Flash Attention
- Fused QKV projection or separate (configurable)
- Flash Attention integration with pair bias
- Fallback to standard attention

**`FusedMSAColumnAttention`**: MSA column attention with QKV fusion + Flash Attention
- Fused QKV projection
- Flash Attention for column-wise operations

**`FusedTriangleMultiplication`**: Triangle update with gate/proj fusion
- Fused left_right_proj (2 ops → 1)
- Fused left_right_gate (2 ops → 1)
- Einstein summation for triangle computation

**`FusedTriangleAttention`**: Triangle attention with QKV fusion + Flash Attention
- Fused QKV projections
- Flash Attention for edge attention

**`FusedEvoformerBlock`**: Complete Evoformer with all fusions
- Integrates all fused components
- Maintains compatibility with baseline architecture

**`TinyOpenFoldV2`**: Main model class with fusion support
- Accepts FusionConfig parameter
- Supports torch.compile wrapper
- Fusion statistics reporting

### Fusion Statistics

```python
# Get fusion statistics from model
fusion_stats = model.get_fusion_statistics()

# Returns:
# {
#     'qkv_fusion_msa_enabled': True,
#     'qkv_fusion_triangle_enabled': True,
#     'flash_attention_enabled': True,
#     'triangle_fusion_enabled': True,
#     'baseline_kernels_per_block': 15,
#     'fused_kernels_per_block': 3,
#     'kernel_reduction_percent': 80.0,
#     'total_kernel_reduction': 48
# }
```

## Debugging Tips

### Fusion Not Working

```bash
# Check Flash Attention availability
python3 -c "import torch.nn.functional as F; print(hasattr(F, 'scaled_dot_product_attention'))"

# Check torch.compile availability
python3 -c "import torch; print(hasattr(torch, 'compile'))"

# Run with fusion disabled to compare
python3 tiny_openfold_v2.py --disable-all-fusion
```

### Numerical Accuracy Verification

```bash
# Verify that fused version produces numerically equivalent outputs to baseline
python3 tiny_openfold_v2.py --verify-accuracy --batch-size 4

# Output shows:
# - Absolute differences (max, mean)
# - Relative differences (max, mean)
# - Numerical equivalence check (PASS/FAIL)
# - Tolerance: rtol=1e-3, atol=1e-4
```

**What it does:**
- Creates both fused and unfused models with identical weights
- Runs inference with the same inputs
- Compares outputs using `torch.allclose()` with tolerance `rtol=1e-3, atol=1e-4`
- Reports absolute and relative differences

**Expected result:** ✓ PASS - Fusion optimizations should produce outputs within numerical precision tolerance

### Performance Debugging

```bash
# Profile with different fusion combinations
python3 tiny_openfold_v2.py --disable-flash-attention --enable-pytorch-profiler
python3 tiny_openfold_v2.py --disable-qkv-fusion-msa --enable-pytorch-profiler

# Compare kernel counts
grep "kernel" pytorch_profiles_v2/fusion_analysis.json
```

## Understanding Fusion Impact

### Key Areas to Study in Code

1. **FusedMSARowAttention** (lines ~276-384)
   - QKV fusion implementation
   - Flash Attention integration with pair bias
   - Fallback to baseline

2. **FusedTriangleMultiplication** (lines ~532-602)
   - Gate/proj fusion technique
   - Chunk operations for splitting
   - Performance comparison points

3. **get_fusion_statistics()** (lines ~873-907)
   - Kernel reduction calculation
   - Fusion effectiveness metrics

4. **Training loop with fusion tracking** (lines ~1106-1175)
   - Fusion statistics collection
   - Performance monitoring integration

## Workshop Exercises

### Exercise 1: Kernel Fusion Analysis

**Objective**: Quantify the impact of kernel fusion on performance.

```bash
# Run baseline (V1 or V2 with fusions disabled)
python3 tiny_openfold_v2.py --disable-all-fusion --batch-size 4 --num-steps 50 \
    --profile-dir ./baseline

# Run with all fusions
python3 tiny_openfold_v2.py --enable-all-fusion --batch-size 4 --num-steps 50 \
    --profile-dir ./fused

# Compare results
diff baseline/performance_summary_v2.json fused/performance_summary_v2.json
```

**Expected Results:**
- 1.5-2.2x speedup in training speed
- 60-80% reduction in major kernel launches
- 50-80% memory reduction with Flash Attention

### Exercise 2: Flash Attention Memory Analysis

**Objective**: Analyze memory efficiency improvements from Flash Attention.

```bash
# Test with Flash Attention disabled
python3 tiny_openfold_v2.py --disable-flash-attention --seq-len 128 \
    --enable-memory-profiling --profile-dir ./no_flash

# Test with Flash Attention enabled
python3 tiny_openfold_v2.py --enable-flash-attention --seq-len 128 \
    --enable-memory-profiling --profile-dir ./with_flash

# Compare peak memory usage
grep "peak_memory_mb" */performance_summary_v2.json
```

**Expected Results:**
- Linear memory scaling with Flash Attention
- 50-80% memory reduction for sequences > 64
- Enables larger batch sizes or sequence lengths

### Exercise 3: ROCm Profiling Deep Dive

**Objective**: Use ROCm tools for hardware-level analysis.

```bash
# rocprofv3 for kernel statistics
./run_rocprofv3.sh --batch-size 4 --seq-len 64

# rocprof-compute for roofline analysis
./run_rocprof_compute.sh --roof-only --batch-size 4

# Compare kernel counts with baseline
# Verify fusion effectiveness at hardware level
```

**Expected Results:**
- Detailed kernel execution times
- Verification of kernel count reduction
- Memory bandwidth improvements

## Next Steps

After mastering Version 2:

1. **Analyze Fusion Impact**
   - Compare profiling results with V1 baseline
   - Identify which fusions provide most benefit
   - Understand trade-offs and limitations

2. **ROCm Profiling Mastery**
   - Learn to interpret roofline plots
   - Identify memory vs compute bound operations
   - Use hardware counters for optimization

3. **Ablation Studies**
   - Test individual fusion contributions
   - Find optimal fusion combinations for your workload
   - Understand fusion interactions

4. **Production Considerations**
   - Apply learnings to real AlphaFold/OpenFold
   - Consider custom kernel implementations (Version 3)
   - Scale to multi-GPU deployments

## Resources

### AlphaFold 2 & OpenFold
- AlphaFold 2 Paper: https://www.nature.com/articles/s41586-021-03819-2
- OpenFold GitHub: https://github.com/aqlaboratory/openfold
- OpenFold Documentation: https://openfold.readthedocs.io/

### Flash Attention
- Flash Attention Paper: https://arxiv.org/abs/2205.14135
- Flash Attention v2: https://arxiv.org/abs/2307.08691
- PyTorch Documentation: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

### ROCm Profiling
- ROCm Documentation: https://rocm.docs.amd.com/
- rocprof-compute Guide: https://rocm.docs.amd.com/projects/rocprofiler-compute/
- AMD GPU Architecture: https://www.amd.com/en/technologies/cdna

### Parent Directory
- See `../ARCHITECTURE.md` for detailed Evoformer architecture
- See `../version1_pytorch_baseline/README.md` for baseline implementation
- See `PLAN.md` for complete implementation roadmap

---

**Questions or Issues?**

Check the comprehensive profiling reports, examine fusion statistics, or compare with the baseline implementation for detailed understanding of each optimization.

