
# Version 2: PyTorch Fused - Kernel Fusion and ROCm Tools Integration

README.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version2_pytorch_fused` in the Training Examples repository

## Overview

Version 2 demonstrates the power of kernel fusion and introduces comprehensive ROCm profiling tools. Building on the baseline analysis from Version 1, this version implements targeted optimizations to achieve significant performance improvements through strategic kernel fusion, Flash Attention, and advanced ROCm profiling integration.

## Learning Objectives

After completing this version, you will be able to:

- Implement QKV fusion to reduce kernel launch overhead
- Integrate Flash Attention for memory-efficient attention computation
- Apply SwiGLU fusion in feed-forward networks
- Use ROCm profiling tools (rocprofv3, rocprof-sys, rocprof-compute) for hardware-level analysis
- Analyze kernel fusion impact on performance and memory usage
- Interpret ROCm profiling data for optimization insights

## Key Optimizations Implemented

### 1. QKV Fusion

- **Problem**: Separate Q, K, V linear projections create 3 kernel launches
- **Solution**: Fused QKV projection with single kernel launch
- **Expected Benefit**: 20-30% reduction in attention overhead

### 2. Flash Attention Integration

- **Problem**: Standard attention has O(n^2) memory complexity
- **Solution**: PyTorch's scaled_dot_product_attention with Flash Attention
- **Expected Benefit**: Significant memory reduction, enables larger sequences

### 3. SwiGLU Fusion

- **Problem**: Separate gate and up projections in feed-forward network
- **Solution**: Combined gate/up computation with element-wise operations
- **Expected Benefit**: 15-25% feed-forward network speedup

### 4. Torch Compile Integration

- **Problem**: Remaining kernel launch overhead
- **Solution**: Automatic fusion through torch.compile()
- **Expected Benefit**: Additional 10-20% speedup through automatic optimizations

## Architecture Enhancements and Fusion Techniques

### Mathematical Foundation of Kernel Fusion

Kernel fusion combines multiple operations into a single GPU kernel to reduce memory bandwidth requirements and kernel launch overhead. For complete mathematical foundations, see [TINY_LLAMA_ARCHITECTURE.md](../TINY_LLAMA_ARCHITECTURE.md).

#### Fusion Efficiency Analysis

**Memory Bandwidth Reduction:**

$$
\text{Bandwidth Reduction} = 1 - \frac{\text{Fused Operations Memory}}{\text{Separate Operations Memory}}
$$

**For QKV Fusion:**

$$
\begin{aligned}
\text{Separate}: & \quad 3 \times (\text{Input Read} + \text{Weight Read} + \text{Output Write}) \\
& = 3 \times (B \times S \times D + D^2 + B \times S \times D) \\
\text{Fused}: & \quad \text{Input Read} + 3 \times \text{Weight Read} + \text{Output Write} \\
& = B \times S \times D + 3 \times D^2 + B \times S \times 3D \\
\text{Reduction}: & \quad \frac{2 \times B \times S \times D}{\text{Total Separate Memory}} \approx 40\% \text{ for typical batch sizes}
\end{aligned}
$$

### 1. QKV Fusion Implementation

#### Detailed QKV Fusion Analysis

**Before Fusion (Baseline):**
```python
# Three separate linear projections - 3 kernel launches
q = self.q_proj(hidden_states)  # Kernel 1: GEMM [B,S,D] × [D,D] = [B,S,D]
k = self.k_proj(hidden_states)  # Kernel 2: GEMM [B,S,D] × [D,D] = [B,S,D]
v = self.v_proj(hidden_states)  # Kernel 3: GEMM [B,S,D] × [D,D] = [B,S,D]

# Memory reads: 3x input tensor + 3x weight matrices
# Memory writes: 3x output tensors
# Total FLOPS: 3 × (2 × B × S × D^2)
```

**After Fusion (Optimized):**
```python
# Single fused projection - 1 kernel launch
qkv = self.qkv_proj(hidden_states)  # Kernel 1: GEMM [B,S,D] × [D,3D] = [B,S,3D]
q, k, v = qkv.chunk(3, dim=-1)       # Tensor view operation (no memory copy)

# Memory reads: 1x input tensor + 1x weight matrix (3x size)
# Memory writes: 1x output tensor (3x size)
# Total FLOPS: 2 × B × S × D × 3D = 6 × B × S × D^2  (same compute)
```

**Performance Analysis:**
```python
# Kernel launch overhead reduction
KERNEL_LAUNCH_OVERHEAD = {
    'baseline_launches': 3,
    'fused_launches': 1,
    'reduction': '67% fewer kernel launches',
    'overhead_per_launch': '5-50 μs depending on operation size',
    'total_overhead_saved': '10-100 μs per attention layer'
}

# Memory bandwidth optimization
MEMORY_BANDWIDTH = {
    'baseline_reads': 'B×S×D (input) × 3 + D^2 × 3 (weights)',
    'fused_reads': 'B×S×D (input) × 1 + D^2 × 3 (weights)',
    'bandwidth_reduction': '~40% for typical batch sizes',
    'cache_efficiency': 'Improved due to temporal locality'
}
```

#### Fused QKV Implementation

```python
class FusedQKVAttention(nn.Module):
    """QKV-fused attention with detailed performance optimizations."""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads

        # Single fused QKV projection - critical optimization!
        self.qkv_proj = nn.Linear(
            config.hidden_dim,
            3 * config.hidden_dim,
            bias=False
        )
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # RoPE for position embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        # OPTIMIZATION 1: Fused QKV projection (3 ops → 1 op)
        with nvtx.range("fused_qkv_projection"):
            qkv = self.qkv_proj(hidden_states)  # [B, S, 3*D]

        # OPTIMIZATION 2: Efficient tensor chunking (no memory copy)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, S, D]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (rotary position embeddings)
        q, k = self.rotary_emb(q, k, seq_len)

        # OPTIMIZATION 3: Flash Attention (covered in next section)
        with nvtx.range("flash_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=True  # Enables causal masking optimization
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        return self.o_proj(attn_output)
```

### 2. Flash Attention Deep Dive

#### Memory Complexity Analysis

**Standard Attention Memory:**

$$
\begin{aligned}
\text{Attention Matrix} &: \mathcal{O}(B \times H \times S^{2}) \\
\text{For } S=1024: &\quad 1024^2 = 1M \text{ elements per head} \\
\text{Total Memory} &: B \times H \times S^{2} \times 4 \text{ bytes} \\
\text{Example}: &\quad 8 \times 8 \times 1024^2 \times 4 = 268\text{MB}
\end{aligned}
$$

**Flash Attention Memory:**

$$
\begin{aligned}
\text{Block Size} &: B_r \times B_c \quad (\text{typically } 64 \times 64) \\
\text{Memory Usage} &: \mathcal{O}(B \times H \times (B_r + B_c) \times \frac{S^{2}}{B_r \times B_c}) \\
&= \mathcal{O}(B \times H \times S) \text{ (linear in sequence length!)} \\
\text{Reduction} &: \frac{S^{2}}{S} = S \text{-fold memory reduction}
\end{aligned}
$$

#### Flash Attention Implementation Details

```python
# Flash Attention Algorithm (PyTorch implementation)
def flash_attention_forward(q, k, v, mask=None):
    """Memory-efficient attention with O(N) memory complexity."""

    # Use PyTorch's optimized implementation
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=True,  # Enables causal mask optimization
        scale=None       # Uses 1/sqrt(head_dim) automatically
    )

# The above function automatically:
# 1. Tiles the computation into blocks
# 2. Computes attention scores incrementally
# 3. Maintains numerical stability with online softmax
# 4. Minimizes memory transfers between HBM and SRAM
```

**Flash Attention Performance Characteristics:**
```python
FLASH_ATTENTION_BENEFITS = {
    'memory_complexity': {
        'standard': 'O(B × H × S^2)',
        'flash': 'O(B × H × S)',
        'reduction_factor': 'S (sequence length)'
    },
    'computation': {
        'flops': 'Same as standard attention',
        'io_complexity': 'O(S^2 / √M) vs O(S^2) where M is SRAM size',
        'wall_clock': '2-4x faster for sequences > 512'
    },
    'numerical_stability': {
        'method': 'Online softmax with running max',
        'precision': 'Better numerical stability than standard attention',
        'overflow_protection': 'Built-in overflow/underflow handling'
    }
}
```

### 3. SwiGLU Fusion Implementation

#### SwiGLU Mathematical Analysis

**Baseline SwiGLU (Separate Operations):**

$$
\begin{aligned}
\text{gate} &= xW_{\text{gate}} + b_{\text{gate}} \quad \text{(Linear projection 1)} \\
\text{up} &= xW_{\text{up}} + b_{\text{up}} \quad \text{(Linear projection 2)} \\
\text{activated} &= \text{SiLU}(\text{gate}) \quad \text{(Activation function)} \\
\text{intermediate} &= \text{activated} \odot \text{up} \quad \text{(Element-wise multiply)} \\
\text{output} &= \text{intermediate} W_{\text{down}} + b_{\text{down}} \quad \text{(Linear projection 3)}
\end{aligned}
$$

**Fused SwiGLU (Optimized):**

$$
\begin{aligned}
\text{gate\_up} &= x[W_{\text{gate}} \parallel W_{\text{up}}] \quad \text{(Single GEMM)} \\
\text{gate, up} &= \text{split}(\text{gate\_up}, \text{dim}=-1) \quad \text{(Tensor view)} \\
\text{output} &= (\text{SiLU}(\text{gate}) \odot \text{up})W_{\text{down}} \quad \text{(Fused activation + projection)}
\end{aligned}
$$

#### Performance Impact Analysis

```python
# FLOP count comparison
SWIGLU_FLOPS = {
    'gate_projection': 2 * batch_size * seq_len * hidden_dim * intermediate_dim,
    'up_projection': 2 * batch_size * seq_len * hidden_dim * intermediate_dim,
    'down_projection': 2 * batch_size * seq_len * intermediate_dim * hidden_dim,
    'silu_activation': batch_size * seq_len * intermediate_dim,  # Element-wise
    'elementwise_multiply': batch_size * seq_len * intermediate_dim,  # Element-wise
}

# Memory access pattern optimization
MEMORY_ACCESS_OPTIMIZATION = {
    'baseline_memory_ops': {
        'gate_proj': 'Input read + Weight read + Output write',
        'up_proj': 'Input read + Weight read + Output write',
        'down_proj': 'Input read + Weight read + Output write',
        'total_input_reads': 3,  # Major inefficiency!
    },
    'fused_memory_ops': {
        'gate_up_proj': 'Input read + Weight read + Output write',
        'down_proj': 'Input read + Weight read + Output write',
        'total_input_reads': 2,  # 33% reduction in memory bandwidth
    }
}
```

#### Detailed SwiGLU Fusion Implementation

```python
class FusedSwiGLU(nn.Module):
    """SwiGLU with gate/up projection fusion for optimal performance."""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim

        # OPTIMIZATION: Fused gate and up projections
        self.gate_up_proj = nn.Linear(
            self.hidden_dim,
            2 * self.intermediate_dim,  # Combined weight matrix
            bias=False
        )

        self.down_proj = nn.Linear(
            self.intermediate_dim,
            self.hidden_dim,
            bias=False
        )

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # OPTIMIZATION 1: Single GEMM for gate and up projections
        with nvtx.range("fused_gate_up_projection"):
            gate_up = self.gate_up_proj(hidden_states)  # [B, S, 2*I]

        # OPTIMIZATION 2: Efficient tensor splitting (no memory copy)
        gate, up = gate_up.chunk(2, dim=-1)  # Each: [B, S, I]

        # OPTIMIZATION 3: Fused SiLU activation with element-wise multiply
        with nvtx.range("silu_and_multiply"):
            # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            intermediate = F.silu(gate) * up

        # Final down projection
        with nvtx.range("down_projection"):
            output = self.down_proj(intermediate)

        return output
```

**Advanced SwiGLU Optimizations:**
```python
# Custom SiLU implementation for maximum efficiency
def fused_silu_multiply(gate, up):
    """Fused SiLU activation with element-wise multiplication."""
    # Can be further optimized with custom kernels in Version 3
    return F.silu(gate) * up

# Memory layout optimization
def optimized_weight_layout(gate_weight, up_weight):
    """Optimize weight matrix layout for fused GEMM."""
    # Concatenate weights for optimal memory access
    return torch.cat([gate_weight, up_weight], dim=0)
```

### 4. Torch Compile Integration

#### Graph-Level Optimization

```python
# Automatic fusion through torch.compile
@torch.compile(mode='max-autotune')
class CompiledTinyLlama(nn.Module):
    """Automatically optimized model with torch.compile."""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            FusedTransformerBlock(config) for _ in range(config.num_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        # torch.compile will automatically:
        # 1. Fuse adjacent operations
        # 2. Optimize memory layouts
        # 3. Generate specialized kernels
        # 4. Eliminate redundant operations

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return self.norm(hidden_states)
```

**Torch Compile Optimization Benefits:**
```python
TORCH_COMPILE_OPTIMIZATIONS = {
    'automatic_fusion': {
        'elementwise_ops': 'Fuses adjacent elementwise operations',
        'reduction_ops': 'Combines reductions where possible',
        'memory_planning': 'Optimizes tensor allocation and deallocation'
    },
    'kernel_specialization': {
        'shape_specialization': 'Generates optimized kernels for specific shapes',
        'dtype_optimization': 'Optimizes for specific data types',
        'device_targeting': 'AMD GPU-specific optimizations'
    },
    'graph_optimization': {
        'dead_code_elimination': 'Removes unused operations',
        'constant_folding': 'Precomputes constant expressions',
        'common_subexpression': 'Eliminates redundant computations'
    }
}
```

### Fusion Performance Analysis Framework

#### Kernel Launch Reduction Analysis

```python
# Theoretical kernel count analysis
KERNEL_COUNT_ANALYSIS = {
    'baseline_attention': {
        'q_projection': 1,
        'k_projection': 1,
        'v_projection': 1,
        'attention_computation': 3,  # QK^T, softmax, attention*V
        'output_projection': 1,
        'total': 7
    },
    'fused_attention': {
        'qkv_projection': 1,  # Fused Q,K,V
        'flash_attention': 1,  # Optimized attention
        'output_projection': 1,
        'total': 3
    },
    'reduction': '57% fewer kernels per attention layer'
}

# Memory bandwidth utilization
MEMORY_BANDWIDTH_ANALYSIS = {
    'baseline_efficiency': {
        'multiple_small_ops': 'Poor memory bandwidth utilization',
        'cache_misses': 'Frequent cache evictions between operations',
        'bandwidth_usage': '40-60% of peak bandwidth'
    },
    'fused_efficiency': {
        'larger_operations': 'Better memory bandwidth utilization',
        'temporal_locality': 'Improved cache reuse',
        'bandwidth_usage': '70-85% of peak bandwidth'
    }
}
```

#### Arithmetic Intensity Optimization

```python
# Roofline model analysis for fusion optimizations
def calculate_arithmetic_intensity(operation_type, batch_size, seq_len, hidden_dim):
    """Calculate arithmetic intensity for roofline analysis."""

    intensity_metrics = {
        'baseline_attention': {
            'flops': 4 * batch_size * seq_len * hidden_dim ** 2,
            'memory_bytes': 3 * (batch_size * seq_len * hidden_dim * 4),  # 3 separate reads
            'arithmetic_intensity': 'flops / memory_bytes'
        },
        'fused_qkv_attention': {
            'flops': 4 * batch_size * seq_len * hidden_dim ** 2,  # Same compute
            'memory_bytes': 1 * (batch_size * seq_len * hidden_dim * 4),  # Single read
            'arithmetic_intensity': '3x higher than baseline'
        }
    }

    return intensity_metrics
```

## Workshop Exercises

### Exercise 1: Kernel Fusion Analysis

**Objective**: Compare baseline vs. fused implementations to quantify fusion benefits.

#### Step 1: Baseline Comparison
```bash
# Run Version 1 baseline for comparison
cd ../version1_pytorch_baseline
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 30 > ../version2_baseline_comparison.log

# Run Version 2 fused implementation
cd ../version2_pytorch_fused
python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 30 > fused_performance.log
```

#### Step 2: Kernel Count Analysis
```bash
# PyTorch profiler comparison
python run_pytorch_profiler.py --batch-size 8 --profile-dir ./fusion_analysis --generate-report

# Compare kernel counts between versions
python analyze_kernel_reduction.py --baseline ../version1_pytorch_baseline/pytorch_profiles --fused ./fusion_analysis
```

**Expected Results:**
- 40-60% reduction in kernel launch count
- 1.4-1.8x speedup in overall training
- Improved GPU utilization metrics

### Exercise 2: Flash Attention Memory Analysis

**Objective**: Analyze memory efficiency improvements from Flash Attention.

#### Step 1: Memory Scaling Test
```bash
# Test memory scaling with sequence length
for seq_len in 128 256 512 1024; do
    python tiny_llama_v2.py \
        --seq-len $seq_len \
        --batch-size 4 \
        --enable-memory-profiling \
        --profile-dir ./flash_attention_seq${seq_len}
done
```

#### Step 2: Memory Bandwidth Analysis
```bash
# Analyze memory bandwidth utilization
python run_deepspeed_flops.py \
    --batch-size 8 \
    --seq-len 256 \
    --computational-intensity \
    --generate-roofline
```

**Expected Results:**

- Linear memory scaling vs. quadratic for baseline
- 2-4x memory reduction for longer sequences
- Improved arithmetic intensity metrics

### Exercise 3: ROCm Tools Deep Dive

**Objective**: Master ROCm profiling tools for hardware-level optimization.

AMD offers three performance profiling tools for ROCm based applications:
`rocprofv3`, `rocprof-sys`, and `rocprof-compute`. For more details about these tools, see 
[Appendix C of the TECHNICAL_APPENDICES.md](https://github.com/amd/HPCTrainingExamples/blob/main/MLExamples/TinyTransformer/TECHNICAL_APPENDICES.md#appendix-c-rocm-profiling-tools-reference).
about each tool. 

#### Step 1: rocprofv3 Basic Profiling

Running rocprofv3 to collect GPU hotspots on this example would look like this:

```bash
rocprofv3 --kernel-trace --stats --truncate-kernels -- python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 30
```

View the `<pid>_kernel_stats.csv` file to see the GPU kernel hotspots.

#### Step 2: rocprof-sys System Analysis

To collect a comprehensive timeline trace with host and device activity, run rocprof-sys as shown below:

```bash
rocprof-sys-run --profile --trace -- python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 30
```

Copy the `.proto` file to your laptop to visualize with the Perfetto browser based tool at [https://ui.perfetto.dev](https://ui.perfetto.dev).

#### Step 3: rocprof-compute Advanced Analysis

To collect roofline plots, run the following command:

```bash
rocprof-compute profile -n roof --kernel-names --roof-only --device 0 -- python tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 30
```

This generates three PDF files: two roofline plots and a legend.

To collect a profile, then analyze a particular dispatch, run the following commands:

```bash
rocprof-compute profile -n ver2 --no-roof -- python3 tiny_llama_v2.py --batch-size 8 --seq-len 128 --num-steps 30
rocprof-compute analyze -p workloads/ver2/MI300A_A1 --list-stats >& stats.txt
rocprof-compute analyze -p workloads/ver2/MI300A_A1 --dispatch 1538 >& dispatch_1538.txt
```

The `--list-stats` option provides a hotspot list of GPU kernels and a list of dispatches. Pick a dispatch of the
kernel that you want to analyze further and use that in the subsequent analyze command. For example, we are
analyzing dispatch 1538 here.

<!--
**Expected Results:**
- Detailed kernel performance metrics
- Memory hierarchy utilization analysis
- Optimization recommendations for Version 3
-->

## Key Performance Improvements

### Expected Performance Gains

| Optimization | Impact | Memory Reduction | Kernel Reduction | Implementation Effort |
|-------------|--------|------------------|------------------|---------------------|
| **QKV Fusion** | 1.2-1.4x | 15-25% | 33% (3→1 kernels) | Low |
| **Flash Attention** | 1.3-2.0x | 50-80% | 20% fewer kernels | Medium |
| **SwiGLU Fusion** | 1.1-1.3x | 10-20% | 50% (2→1 kernels) | Low |
| **Torch Compile** | 1.1-1.2x | 5-10% | 10-30% | Very Low |
| **Combined Effect** | **1.6-2.5x** | **60-90%** | **40-60%** | - |

### Scaling Characteristics

- **Batch Size Scaling**: Improved efficiency at larger batch sizes
- **Sequence Length Scaling**: Near-linear memory scaling (vs. quadratic)
- **Model Size Scaling**: Better utilization for larger hidden dimensions
- **Multi-GPU Scaling**: Reduced communication overhead

<!-- Commenting this until we can fix scripts

## ROCm Profiling Workflow

### Comprehensive Profiling Pipeline

```bash
# Complete ROCm profiling suite
bash run_all_profilers.sh \
    --batch-size 8 \
    --seq-len 128 \
    --enable-rocm-tools \
    --profile-dir ./complete_rocm_analysis
```

This orchestrates:
1. **PyTorch Profiler** - Framework-level analysis
2. **DeepSpeed FLOPS** - Computational efficiency
3. **rocprofv3** - Basic kernel profiling
4. **rocprof-sys** - System monitoring
5. **rocprof-compute** - Advanced analysis

### Profiling Data Analysis

```bash
# Generate comprehensive comparison report
python generate_fusion_report.py \
    --baseline ../version1_pytorch_baseline/complete_analysis \
    --fused ./complete_rocm_analysis \
    --output ./fusion_comparison_report.md
```
-->

## Advanced Features

### Configurable Fusion Levels

```bash
# Selective fusion testing
python tiny_llama_v2.py \
    --enable-qkv-fusion \
    --enable-flash-attention \
    --disable-swiglu-fusion \
    --enable-torch-compile

# A/B testing different fusion combinations
python fusion_ablation_study.py --all-combinations
```

### Dynamic Batch Size Optimization

```bash
# Find optimal batch size for current hardware
python optimize_batch_size.py \
    --target-memory-usage 0.8 \
    --seq-len 128 \
    --optimization-target throughput
```

### Mixed Precision Integration

```bash
# Test mixed precision with fusion
python tiny_llama_v2.py \
    --use-amp \
    --amp-dtype bfloat16 \
    --enable-all-fusion
```

## Performance Validation

### Regression Testing

```bash
# Numerical accuracy validation
python validate_numerical_accuracy.py \
    --baseline ../version1_pytorch_baseline/tiny_llama_v1.py \
    --optimized ./tiny_llama_v2.py \
    --tolerance 1e-4

# Performance regression testing
python performance_regression_test.py \
    --baseline-results ../version1_baseline_metrics.json \
    --current-results ./version2_metrics.json \
    --min-speedup 1.3
```

### Benchmark Suite

```bash
# Comprehensive benchmarking
python benchmark_suite.py \
    --models v1,v2 \
    --batch-sizes 4,8,16,32 \
    --seq-lengths 128,256,512 \
    --metrics throughput,memory,accuracy
```

## Troubleshooting

### Common Issues

#### Flash Attention Compatibility
```bash
# Check PyTorch version compatibility
python -c "import torch; print(torch.__version__); print(hasattr(torch.nn.functional, 'scaled_dot_product_attention'))"

# Fallback for older PyTorch versions
export PYTORCH_FALLBACK_ATTENTION=1
```

#### ROCm Tools Permission Issues
```bash
# Ensure proper permissions for ROCm profiling
sudo usermod -a -G render $USER
export ROCPROF_COMPUTE_DISABLE_AQL_DEBUG=1
```

#### Memory Issues with Larger Sequences
```bash
# Enable memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export HIP_LAUNCH_BLOCKING=1  # For debugging
```

<!-- 
### Performance Debugging

#### Kernel Launch Analysis
```bash
# Analyze kernel launch patterns
rocprof --hip-trace --kernel-trace python tiny_llama_v2.py
python analyze_kernel_launches.py --trace-file results.csv
```

#### Memory Bandwidth Utilization
```bash
# Detailed memory analysis
python memory_bandwidth_analyzer.py \
    --profile-data ./complete_rocm_analysis \
    --generate-roofline-plot
```
-->

## Expected Learning Outcomes

### Technical Skills Developed

- **Kernel Fusion Techniques**: Practical implementation of operation fusion
- **Memory Optimization**: Understanding memory-efficient algorithm design
- **ROCm Profiling Mastery**: Comprehensive hardware profiling skills
- **Performance Analysis**: Data-driven optimization decision making

### Performance Engineering Insights

- **Amdahl's Law in Practice**: Understanding optimization impact distribution
- **Memory vs. Compute Trade-offs**: Balancing different optimization strategies
- **Hardware Utilization**: Maximizing GPU resource utilization
- **Scaling Characteristics**: How optimizations affect different workload sizes

## Next Steps

After mastering Version 2:

1. **Analyze fusion impact** across different model and batch configurations
2. **Identify remaining bottlenecks** using ROCm profiling data
3. **Prepare optimization targets** for Version 3 (Triton kernels)
4. **Document lessons learned** for production deployment
5. **Establish performance baselines** for advanced optimizations

**Ready for Custom Kernels? Proceed to [Version 3: Triton Integration](../version3_triton/README.md)**

<!--
---

## Quick Start Commands

```bash
# Complete Version 2 workflow
cd version2_pytorch_fused

# 1. Basic fused training
python tiny_llama_v2.py --batch-size 8 --enable-all-fusion

# 2. Comprehensive profiling
bash run_all_profilers.sh --enable-rocm-tools

# 3. Compare with Version 1
python compare_versions.py --v1 ../version1_pytorch_baseline --v2 .

# 4. Generate optimization report
python generate_fusion_report.py --output-dir ./optimization_analysis
```
-->

**Expected Results**: 1.6-2.5x speedup, 60-90% memory reduction, comprehensive ROCm profiling mastery.


