
# Version 3: Triton Kernel Integration

README.md from `HPCTrainingExamples/MLExamples/TinyTransformer/version3_triton` in the Training Examples repository

**Objective**: Implement custom GPU kernels using Triton for maximum performance optimization

**Expected Performance**: 2.0-3.5x speedup over baseline, 70-95% memory reduction

**Learning Focus**: GPU kernel programming, memory access optimization, custom operator development

## Overview

Version 3 introduces custom Triton GPU kernels for the most performance-critical operations in the Tiny LLaMA model. Triton provides a Python-like syntax for writing GPU kernels while automatically handling low-level optimizations like memory coalescing and register allocation.

### Key Optimizations

1. **Custom RMSNorm Kernel**: Fused variance computation and normalization
2. **SwiGLU Kernel**: Combined gate/up projections with SiLU activation
3. **Flash Attention Kernel**: Memory-efficient attention with O(N) complexity
4. **Automatic Optimization**: Triton compiler optimizations for target hardware

### Architecture Changes

```
Previous: PyTorch Operations → Multiple Kernel Launches → Memory Transfers
Current:  Custom Triton Kernels → Single Optimized Launch → Minimal Memory Traffic
```

## Files and Structure

```
version3_triton/
├── README.md                           # This file
├── tiny_llama_v3.py                   # Main model with Triton kernels
├── run_triton_profiling.py            # Triton-specific profiling
├── run_rocprof_triton.sh              # ROCProfiler for Triton kernels
├── exercises/
│   ├── exercise1_triton_basics.md     # Triton fundamentals
│   ├── exercise2_swiglu_optimization.md # SwiGLU kernel deep dive
│   └── exercise3_flash_attention.md   # Flash Attention implementation
└── results/                           # Generated profiling results
```

## Key Components and Triton Kernel Implementation

### Mathematical Foundation of Triton Kernels

Triton kernels optimize GPU computation by exploiting the memory hierarchy and parallelism patterns. For complete mathematical foundations, see [TINY_LLAMA_ARCHITECTURE.md](../TINY_LLAMA_ARCHITECTURE.md).

#### Memory Hierarchy Optimization

**GPU Memory Hierarchy:**
```
Registers (fastest, ~40KB per SM)     → Data reuse within thread
Shared Memory (~164KB per SM)         → Data sharing within thread block
L1 Cache (~128KB per SM)              → Automatic caching
L2 Cache (~8MB global)                → Cross-SM data sharing
HBM (slowest, ~64GB)                  → Main memory
```

**Triton Optimization Strategy:**

$$\text{Arithmetic Intensity} = \frac{\text{FLOPS}}{\text{Memory Bytes Accessed}}$$

Triton maximizes this ratio by:

1. **Tiling**: Processing data in blocks that fit in fast memory
2. **Fusion**: Combining multiple operations to reuse data
3. **Vectorization**: Using SIMD instructions efficiently

### 1. Triton RMSNorm Implementation

#### RMSNorm Mathematical Analysis

**Standard Implementation (PyTorch):**
```python
# Multiple kernel launches and memory accesses
variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)  # Kernel 1: Power + Reduction
rstd = torch.rsqrt(variance + eps)                            # Kernel 2: Reciprocal sqrt
output = (x * rstd).to(input_dtype) * weight                  # Kernel 3: Multiply + Scale

# Total: 3 kernel launches, 3x memory bandwidth usage
```

**Triton Fused Implementation:**
```python
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused RMSNorm kernel with optimal memory access patterns.

    Mathematical Operation:
    output = (x / sqrt(mean(x^2) + eps)) * weight

    Memory Optimization:
    - Single pass through input data
    - Variance computation in registers
    - Immediate normalization and scaling
    """
    # Program ID determines which row this thread block processes
    row_idx = tl.program_id(0)

    # Bounds checking
    if row_idx >= n_rows:
        return

    # Compute memory offsets for this row
    x_row_ptr = x_ptr + row_idx * n_cols
    output_row_ptr = output_ptr + row_idx * n_cols

    # Load weight vector (broadcast across all rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)

    # OPTIMIZATION 1: Streaming variance computation
    variance = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input block
        x_block = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)

        # Accumulate variance in registers (no memory writes!)
        variance += tl.sum(x_block * x_block)

    # Compute RMS normalization factor
    variance = variance / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)

    # OPTIMIZATION 2: Fused normalization and scaling
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input block again (cached in L1/L2)
        x_block = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)

        # Fused normalize + scale in single operation
        output_block = x_block * rstd * weight_block

        # Store result
        tl.store(output_row_ptr + col_offsets, output_block, mask=mask)
```

**Performance Analysis:**
```python
RMSNORM_PERFORMANCE = {
    'memory_access_pattern': {
        'pytorch': 'Multiple passes through data',
        'triton': 'Two passes (variance + normalize)',
        'bandwidth_reduction': '~50% fewer memory accesses'
    },
    'kernel_launches': {
        'pytorch': 3,  # pow, mean, multiply
        'triton': 1,   # fused operation
        'overhead_reduction': '67% fewer kernel launches'
    },
    'numerical_precision': {
        'pytorch': 'Multiple intermediate tensors',
        'triton': 'High-precision accumulation in registers',
        'stability': 'Better numerical stability'
    }
}
```

### 2. Triton SwiGLU Implementation

#### SwiGLU Fusion Analysis

**Memory Access Pattern Optimization:**

$$\begin{aligned}
\text{Standard SwiGLU}: & \quad \text{4 separate operations} \\
\text{gate} &= xW_{\text{gate}} \quad \text{(GEMM 1)} \\
\text{up} &= xW_{\text{up}} \quad \text{(GEMM 2)} \\
\text{activated} &= \text{SiLU}(\text{gate}) \quad \text{(Elementwise 1)} \\
\text{output} &= \text{activated} \odot \text{up} \quad \text{(Elementwise 2)} \\
\text{Memory Reads}: & \quad 4 \times \text{input tensor} + 2 \times \text{weight matrices}
\end{aligned}$$

**Triton Fused SwiGLU:**

$$\begin{aligned}
\text{Triton SwiGLU}: & \quad \text{Single fused operation} \\
\text{output} &= \text{SiLU}(xW_{\text{gate}}) \odot (xW_{\text{up}}) \\
\text{Memory Reads}: & \quad 1 \times \text{input tensor} + 2 \times \text{weight matrices}
\end{aligned}$$

#### Detailed Triton SwiGLU Kernel

```python
@triton.jit
def swiglu_kernel(
    x_ptr, gate_weight_ptr, up_weight_ptr, output_ptr,
    batch_size, seq_len, hidden_dim, intermediate_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Fused SwiGLU kernel with optimal memory tiling.

    Computes: output = SiLU(x @ gate_weight) * (x @ up_weight)

    Tiling Strategy:
    - M dimension: batch_size * seq_len
    - K dimension: hidden_dim
    - N dimension: intermediate_dim
    """
    # Thread block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulators for both gate and up projections
    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # OPTIMIZATION 1: Fused GEMM computation
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)

        # Load input tile (shared between gate and up computations)
        x_tile = tl.load(
            x_ptr + m_offset[:, None] * hidden_dim + k_offset[None, :],
            mask=(m_offset[:, None] < batch_size * seq_len) & (k_offset[None, :] < hidden_dim)
        )

        # Load weight tiles
        gate_weight_tile = tl.load(
            gate_weight_ptr + k_offset[:, None] * intermediate_dim + n_offset[None, :],
            mask=(k_offset[:, None] < hidden_dim) & (n_offset[None, :] < intermediate_dim)
        )
        up_weight_tile = tl.load(
            up_weight_ptr + k_offset[:, None] * intermediate_dim + n_offset[None, :],
            mask=(k_offset[:, None] < hidden_dim) & (n_offset[None, :] < intermediate_dim)
        )

        # Fused matrix multiplication (data reuse in registers)
        gate_acc += tl.dot(x_tile, gate_weight_tile)
        up_acc += tl.dot(x_tile, up_weight_tile)

    # OPTIMIZATION 2: Fused SiLU activation and element-wise multiply
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate_activated = gate_acc / (1.0 + tl.exp(-gate_acc))
    swiglu_output = gate_activated * up_acc

    # Store final result
    output_mask = (m_offset[:, None] < batch_size * seq_len) & (n_offset[None, :] < intermediate_dim)
    tl.store(
        output_ptr + m_offset[:, None] * intermediate_dim + n_offset[None, :],
        swiglu_output,
        mask=output_mask
    )
```

**Triton SwiGLU Performance Characteristics:**
```python
SWIGLU_TRITON_BENEFITS = {
    'memory_efficiency': {
        'data_reuse': 'Input tensor loaded once, used for both gate and up',
        'register_usage': 'Intermediate results kept in registers',
        'bandwidth_reduction': '60-75% reduction in memory traffic'
    },
    'computational_efficiency': {
        'operation_fusion': 'GEMM + SiLU + elementwise in single kernel',
        'vectorization': 'Automatic SIMD instruction generation',
        'occupancy': 'Optimized thread block configuration'
    },
    'numerical_stability': {
        'precision': 'FP32 accumulation with FP16 storage',
        'activation_stability': 'Numerically stable SiLU implementation',
        'overflow_protection': 'Built-in overflow handling'
    }
}
```

### 3. Triton Flash Attention Implementation

#### Flash Attention Tiling Strategy

**Memory Complexity Analysis:**

$$\begin{aligned}
\text{Standard Attention Memory} &: O(B \times H \times S^{2}) \\
\text{Flash Attention Memory} &: O(B \times H \times S) \\
\text{SRAM Usage} &: O(B_r + B_c) \text{ where } B_r, B_c \text{ are tile sizes} \\
\text{IO Complexity} &: O\left(\frac{S^{2}}{\sqrt{M}}\right) \text{ where } M \text{ is SRAM size}
\end{aligned}$$

#### Triton Flash Attention Kernel

```python
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Memory-efficient Flash Attention with tiled computation.

    Algorithm:
    1. Tile Q, K, V into blocks that fit in SRAM
    2. Compute attention scores incrementally
    3. Use online softmax for numerical stability
    4. Accumulate attention output progressively
    """
    # Thread block IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_tile_idx = tl.program_id(2)

    # Compute base pointers for this batch and head
    q_base = q_ptr + batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    k_base = k_ptr + batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    v_base = v_ptr + batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim
    output_base = output_ptr + batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim

    # Load Q tile (stays in SRAM for entire computation)
    q_offset_m = q_tile_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    q_mask_m = q_offset_m < seq_len

    q_tile = tl.load(
        q_base + q_offset_m[:, None] * head_dim + tl.arange(0, head_dim)[None, :],
        mask=q_mask_m[:, None]
    )

    # Initialize output accumulator and normalization factors
    output_acc = tl.zeros((BLOCK_SIZE_M, head_dim), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE_M,), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # OPTIMIZATION 1: Tiled computation over K, V
    for k_tile_idx in range(0, tl.cdiv(seq_len, BLOCK_SIZE_N)):
        k_offset_n = k_tile_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k_mask_n = k_offset_n < seq_len

        # Load K and V tiles
        k_tile = tl.load(
            k_base + k_offset_n[:, None] * head_dim + tl.arange(0, head_dim)[None, :],
            mask=k_mask_n[:, None]
        )
        v_tile = tl.load(
            v_base + k_offset_n[:, None] * head_dim + tl.arange(0, head_dim)[None, :],
            mask=k_mask_n[:, None]
        )

        # OPTIMIZATION 2: Compute attention scores in tiles
        scores = tl.dot(q_tile, k_tile.T) * (1.0 / tl.sqrt(head_dim.to(tl.float32)))

        # Apply causal mask
        causal_mask = q_offset_m[:, None] >= k_offset_n[None, :]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # OPTIMIZATION 3: Online softmax (numerically stable)
        tile_max = tl.max(scores, axis=1)
        new_row_max = tl.maximum(row_max, tile_max)

        # Rescale previous accumulated values
        old_scale = tl.exp(row_max - new_row_max)
        tile_scale = tl.exp(tile_max - new_row_max)

        # Update output accumulator
        output_acc = output_acc * old_scale[:, None]
        scores_softmax = tl.exp(scores - new_row_max[:, None]) * tile_scale[:, None]
        output_acc += tl.dot(scores_softmax, v_tile)

        # Update normalization factors
        row_sum = row_sum * old_scale + tl.sum(scores_softmax, axis=1)
        row_max = new_row_max

    # Final normalization
    output_final = output_acc / row_sum[:, None]

    # Store result
    tl.store(
        output_base + q_offset_m[:, None] * head_dim + tl.arange(0, head_dim)[None, :],
        output_final,
        mask=q_mask_m[:, None]
    )
```

**Flash Attention Performance Benefits:**
```python
FLASH_ATTENTION_TRITON = {
    'memory_efficiency': {
        'complexity': 'O(N) vs O(N^2) for standard attention',
        'sram_usage': 'Optimal SRAM utilization with tiling',
        'hbm_access': 'Minimized high-bandwidth memory access'
    },
    'computational_efficiency': {
        'online_softmax': 'Numerically stable incremental computation',
        'tiled_gemm': 'Optimal matrix multiplication blocking',
        'kernel_fusion': 'Single kernel for entire attention computation'
    },
    'scalability': {
        'sequence_length': 'Linear scaling with sequence length',
        'batch_processing': 'Efficient batched computation',
        'multi_head': 'Parallelized across attention heads'
    }
}
```

### Advanced Triton Optimization Techniques

#### Block Size Tuning

```python
def auto_tune_block_sizes(operation_type, input_shape, device_properties):
    """
    Automatically tune block sizes for optimal performance.
    """
    tuning_space = {
        'rmsnorm': {
            'block_sizes': [64, 128, 256, 512, 1024],
            'criteria': 'Memory bandwidth utilization',
            'constraints': 'Register usage < 64KB'
        },
        'swiglu': {
            'block_sizes': [(32, 64, 32), (64, 64, 64), (128, 32, 64)],
            'criteria': 'Arithmetic intensity maximization',
            'constraints': 'Shared memory < 164KB'
        },
        'flash_attention': {
            'block_sizes': [(64, 64), (128, 64), (64, 128)],
            'criteria': 'SRAM utilization efficiency',
            'constraints': 'Memory coalescing requirements'
        }
    }

    return optimize_for_hardware(tuning_space[operation_type], device_properties)
```

#### Memory Coalescing Optimization

```python
# Optimal memory access patterns for AMD GPUs
MEMORY_ACCESS_PATTERNS = {
    'coalesced_access': {
        'pattern': 'Consecutive threads access consecutive memory addresses',
        'bandwidth': '100% of peak memory bandwidth',
        'implementation': 'Proper stride patterns in Triton kernels'
    },
    'strided_access': {
        'pattern': 'Regular stride pattern across memory',
        'bandwidth': '50-80% of peak memory bandwidth',
        'optimization': 'Adjust block sizes to match stride'
    },
    'random_access': {
        'pattern': 'Irregular memory access pattern',
        'bandwidth': '10-30% of peak memory bandwidth',
        'mitigation': 'Data reordering and blocking strategies'
    }
}
```

## Quick Start

### 1. Environment Setup

Ensure Triton is installed in your environment:

```bash
# Should already be installed from setup/
pip install triton
```

Verify Triton installation:

```python
import triton
print(f"Triton version: {triton.__version__}")
```

### 2. Run the Model

Execute the optimized model:

```bash
cd version3_triton/
python3 tiny_llama_v3.py
```

**Expected Output:**
```
=== Triton Kernel Model Benchmark ===
Model size: XXX.X M parameters
Input shape: torch.Size([4, 512])
Average forward pass time: XX.XX ms
Throughput: XXXX tokens/second
Memory allocated: X.XX GB
Estimated FLOPS/second: XX.XX TFLOPS
```

### 3. Profile Performance

Run comprehensive profiling:

```bash
# Triton-specific profiling
python3 run_triton_profiling.py
```

<!--
```
# ROCProfiler analysis
chmod +x run_rocprof_triton.sh
./run_rocprof_triton.sh
```
-->

### 4. Analyze Results

Check generated results:

```bash
ls profiling_results/
cat profiling_results/triton_summary_report.md
```

<!--
```
ls rocprof_results/
cat rocprof_results/triton_analysis_summary.md
```

## Performance Analysis

### Expected Performance Gains

| Component | Baseline Time | Version 2 Time | Version 3 Time | V3 Speedup | V3 vs V2 |
|-----------|---------------|----------------|----------------|------------|----------|
| RMSNorm | 100% | 60-70% | 35-45% | 2.2-2.9x | 1.4-1.9x |
| SwiGLU | 100% | 40-60% | 25-35% | 2.9-4.0x | 1.4-2.0x |
| Attention | 100% | 60-80% | 30-50% | 2.0-3.3x | 1.6-2.3x |
| **Overall** | **100%** | **50-70%** | **30-45%** | **2.2-3.3x** | **1.4-2.0x** |

### Memory Efficiency

| Metric | Standard PyTorch | Version 2 Fused | Version 3 Triton | Improvement |
|--------|------------------|-----------------|------------------|-------------|
| Peak Memory | 100% | 40-60% | 20-35% | 65-80% reduction |
| Memory Bandwidth | 100% | 70-85% | 45-65% | 35-55% reduction |
| Kernel Launches | 100% | 30-50% | 15-25% | 75-85% reduction |

### Triton Kernel Development Workflow

#### Performance Profiling Integration

```python
# Triton kernel profiling with ROCm tools
def profile_triton_kernel(kernel_func, inputs, kernel_name):
    """
    Comprehensive profiling of Triton kernels.
    """
    profiling_config = {
        'kernel_timing': 'Per-kernel execution time',
        'memory_bandwidth': 'Achieved vs theoretical bandwidth',
        'occupancy': 'SM utilization percentage',
        'register_usage': 'Register file utilization',
        'shared_memory': 'Shared memory bank conflicts'
    }

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        # Run kernel multiple times for accurate timing
        for _ in range(100):
            output = kernel_func(*inputs)
            torch.cuda.synchronize()

    return analyze_kernel_performance(prof, kernel_name)
```

#### Kernel Optimization Checklist

```python
TRITON_OPTIMIZATION_CHECKLIST = {
    'memory_optimization': {
        'coalesced_access': 'Ensure consecutive memory access patterns',
        'shared_memory_usage': 'Maximize shared memory utilization',
        'register_spilling': 'Minimize register pressure',
        'cache_efficiency': 'Optimize for L1/L2 cache behavior'
    },
    'compute_optimization': {
        'arithmetic_intensity': 'Maximize FLOPS per byte ratio',
        'vectorization': 'Use SIMD instructions effectively',
        'occupancy': 'Balance thread blocks vs resources',
        'load_balancing': 'Ensure even work distribution'
    },
    'compiler_optimization': {
        'block_size_tuning': 'Auto-tune for target hardware',
        'loop_unrolling': 'Optimize inner loops',
        'constant_folding': 'Precompute constant expressions',
        'dead_code_elimination': 'Remove unused computations'
    }
}
```

## Hands-on Exercises

Work through the exercises in order to build understanding:

### Exercise 1: Triton Basics (45 minutes)

- Understand Triton kernel structure
- Analyze memory access patterns
- Optimize block sizes
- Compare with PyTorch implementations

### Exercise 2: SwiGLU Optimization (60 minutes)

- Multi-dimensional kernel programming
- Arithmetic intensity analysis
- Memory layout optimization
- Advanced kernel variants

### Exercise 3: Flash Attention (75 minutes)

- Memory-efficient attention algorithms
- Tiling strategies and optimization
- Numerical stability considerations
- Large sequence handling

## Advanced Topics

### Kernel Optimization Strategies

1. **Block Size Tuning**
   - Match hardware characteristics
   - Optimize for occupancy
   - Consider memory coalescing

2. **Memory Access Patterns**
   - Minimize global memory access
   - Maximize register usage
   - Optimize cache utilization

3. **Arithmetic Intensity**
   - Balance compute vs memory operations
   - Identify bottlenecks (compute vs memory bound)
   - Apply roofline model analysis

### Debugging Triton Kernels

1. **Compilation Issues**
   - Check tensor shapes and types
   - Verify constexpr usage
   - Review block size constraints

2. **Performance Problems**
   - Profile memory access patterns
   - Check occupancy metrics
   - Analyze kernel launch overhead

3. **Numerical Issues**
   - Monitor for overflow/underflow
   - Check reduction accuracy
   - Verify mask applications

## Integration with ROCm Tools

<!--
### ROCProfiler Analysis

The provided scripts integrate with ROCm profiling tools:

```bash
# Basic kernel profiling
rocprof --stats --kernel-trace python3 tiny_llama_v3.py

# Memory analysis
rocprof --hip-trace --memory-trace python3 tiny_llama_v3.py

# Detailed metrics
rocprof --input kernel_metrics.txt python3 tiny_llama_v3.py
```
-->

### Key Metrics to Monitor

1. **Kernel Performance**
   - Execution time per kernel
   - Launch overhead
   - Occupancy rates

2. **Memory Utilization**
   - Bandwidth efficiency
   - Cache hit rates
   - Memory access patterns

3. **Compute Efficiency**
   - VALU utilization
   - Arithmetic intensity
   - Roofline performance

## Troubleshooting

### Common Issues

1. **Triton Not Found**
   ```bash
   pip install triton
   # Or check environment setup
   ```

2. **Kernel Compilation Errors**
   - Verify GPU compatibility
   - Check CUDA/ROCm installation
   - Review tensor dimensions

3. **Performance Regression**
   - Ensure proper warmup
   - Check block size settings
   - Verify input data layout

4. **Memory Errors**
   - Reduce batch size or sequence length
   - Check for memory leaks
   - Monitor peak memory usage

### Performance Debugging

1. **Profile Each Kernel Individually**
   ```python
   # Isolate kernel performance
   triton_rmsnorm = TritonRMSNorm(dim)
   # Benchmark just this component
   ```

2. **Compare Block Sizes**
   ```python
   # Test different configurations
   for block_size in [64, 128, 256, 512]:
       # Measure performance
   ```

3. **Memory Pattern Analysis**
   ```python
   # Check memory access efficiency
   torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
   ```

## Next Steps

After completing Version 3:

1. **Review Performance Gains**: Compare with previous versions
2. **Understand Optimization Principles**: Kernel design patterns
3. **Prepare for Version 4**: Ultra-fused implementations

Version 4 will combine all optimizations into ultra-fused kernels that process entire transformer blocks in minimal kernel launches.

## Resources

### Documentation
- [Triton Language Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [GPU Architecture Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html)
- [ROCm Profiler Documentation](https://rocmdocs.amd.com/en/latest/ROCm_Tools/ROCm-Tools.html)

### Papers and References
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Triton: A Language for AI Kernel Programming](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [Roofline Model for GPU Performance](https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/)

### AMD ROCm Resources
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- [Performance Optimization Tips](https://rocmdocs.amd.com/en/latest/Programming_Guides/Opencl-programming-guide.html)

