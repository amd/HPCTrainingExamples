# Version 3: Triton Kernel Integration for TinyOpenFold

**Objective**: Implement custom GPU kernels using Triton for maximum performance optimization of the Evoformer architecture

**Expected Performance**: 2.0-3.0x speedup over baseline, 50-70% memory reduction

**Learning Focus**: GPU kernel programming, memory access optimization, Flash Attention for protein structure prediction

## Overview

Version 3 introduces custom Triton GPU kernels for the most performance-critical operations in the Tiny OpenFold model. Triton provides a Python-like syntax for writing GPU kernels while automatically handling low-level optimizations like memory coalescing and register allocation.

### Key Optimizations

1. **Custom LayerNorm Kernel**: Fused mean/variance computation and normalization
2. **Flash Attention for MSA**: Memory-efficient row and column attention with O(N) complexity
3. **Flash Attention for Triangles**: Tiled attention for pair representation updates
4. **Hybrid Optimization**: Triton for memory-bound, PyTorch/rocBLAS for compute-bound operations

### Architecture Changes

```
Previous: PyTorch Operations → Multiple Kernel Launches → Memory Transfers
Current:  Custom Triton Kernels → Single Optimized Launch → Minimal Memory Traffic
```

## Files and Structure

```
version3_triton/
├── README.md                           # This file
├── tiny_openfold_v3.py                 # Main model with Triton kernels
├── run_triton_profiling.py            # Triton-specific profiling
├── run_rocprof_triton.sh              # ROCProfiler for Triton kernels
└── launch_performance_study.sh         # Performance comparison script
```

## Key Components and Triton Kernel Implementation

### Mathematical Foundation of Triton Kernels

Triton kernels optimize GPU computation by exploiting the memory hierarchy and parallelism patterns. For complete Evoformer architecture details, see [../ARCHITECTURE.md](../ARCHITECTURE.md).

#### Memory Hierarchy Optimization

**GPU Memory Hierarchy:**
```
Registers (fastest, ~40KB per SM)     → Data reuse within thread
Shared Memory (~164KB per SM)         → Data sharing within thread block
L1 Cache (~128KB per SM)              → Automatic caching
L2 Cache (~8MB global)                → Cross-SM data sharing
HBM (slowest, ~192GB on MI300X)      → Main memory
```

**Triton Optimization Strategy:**

$$\text{Arithmetic Intensity} = \frac{\text{FLOPS}}{\text{Memory Bytes Accessed}}$$

Triton maximizes this ratio by:

1. **Tiling**: Processing data in blocks that fit in fast memory
2. **Fusion**: Combining multiple operations to reuse data
3. **Vectorization**: Using SIMD instructions efficiently

### 1. Triton LayerNorm Implementation

#### LayerNorm Mathematical Analysis

**Standard Implementation (PyTorch):**
```python
# Multiple kernel launches and memory accesses
mean = x.mean(-1, keepdim=True)                           # Kernel 1: Reduction
variance = ((x - mean) ** 2).mean(-1, keepdim=True)       # Kernel 2: Power + Reduction
output = (x - mean) / torch.sqrt(variance + eps) * weight # Kernel 3: Normalize + Scale

# Total: 3+ kernel launches, 4+ passes through data
```

**Triton Fused Implementation:**
```python
@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Fused LayerNorm kernel with optimal memory access patterns.

    Mathematical Operation:
    output = (x - mean) / sqrt(variance + eps) * weight

    Memory Optimization:
    - Two passes through input data (statistics + normalize)
    - Mean and variance computed in registers
    - Immediate normalization and scaling
    """
    row_idx = tl.program_id(0)

    # Pass 1: Compute mean
    mean = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_vals = tl.load(x_ptr + row_idx * n_elements + offsets, mask=mask, other=0.0)
        mean += tl.sum(x_vals, axis=0)
    mean = mean / n_elements

    # Pass 2: Compute variance
    variance = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_vals = tl.load(x_ptr + row_idx * n_elements + offsets, mask=mask, other=0.0)
        variance += tl.sum((x_vals - mean) * (x_vals - mean), axis=0)
    variance = variance / n_elements
    inv_std = 1.0 / tl.sqrt(variance + eps)

    # Pass 3: Normalize and scale (fused)
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_vals = tl.load(x_ptr + row_idx * n_elements + offsets, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        normalized = (x_vals - mean) * inv_std * weight_vals
        tl.store(output_ptr + row_idx * n_elements + offsets, normalized, mask=mask)
```

**Performance Analysis:**
```python
LAYERNORM_PERFORMANCE = {
    'memory_access_pattern': {
        'pytorch': 'Multiple separate passes through data',
        'triton': 'Three optimized passes (mean, variance, normalize)',
        'bandwidth_reduction': '~40% fewer memory accesses'
    },
    'kernel_launches': {
        'pytorch': 3,  # mean, variance, normalize
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

### 2. Flash Attention for MSA Operations

#### MSA Attention Complexity Analysis

**Standard Attention Memory:**

$$\begin{aligned}
\text{Memory for Scores} &: O(B \times N_{seqs} \times N_{res}^{2} \times H) \\
\text{Standard Attention} &: \text{Materialize full attention matrix} \\
\text{Flash Attention} &: O(B \times N_{seqs} \times N_{res} \times H)
\end{aligned}$$

Where:
- $B$ = batch size
- $N_{seqs}$ = number of MSA sequences (16)
- $N_{res}$ = sequence length (64 residues)
- $H$ = number of heads (4)

#### Triton Flash Attention Kernel

```python
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim, scale,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """
    Memory-efficient Flash Attention with tiled computation.

    Algorithm:
    1. Tile Q, K, V into blocks that fit in SRAM
    2. Compute attention scores incrementally
    3. Use online softmax for numerical stability
    4. Accumulate attention output progressively

    Memory Complexity: O(N) vs O(N²) for standard attention
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)

    # Calculate base offset for this batch/head
    head_offset = batch_idx * num_heads * seq_len * HEAD_DIM + head_idx * seq_len * HEAD_DIM

    # Load Q block (stays in SRAM for entire computation)
    q_start = q_block_idx * BLOCK_SIZE_Q
    q_range = tl.arange(0, BLOCK_SIZE_Q)
    d_range = tl.arange(0, HEAD_DIM)
    
    q_offsets = head_offset + (q_start + q_range[:, None]) * HEAD_DIM + d_range[None, :]
    q_mask = (q_start + q_range[:, None]) < seq_len
    q_block = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

    # Initialize output accumulator and normalization factors
    output_acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    max_scores = tl.full((BLOCK_SIZE_Q,), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)

    # OPTIMIZATION: Tiled computation over K, V
    num_k_blocks = tl.cdiv(seq_len, BLOCK_SIZE_K)
    for k_block_idx in range(num_k_blocks):
        k_start = k_block_idx * BLOCK_SIZE_K
        k_range = tl.arange(0, BLOCK_SIZE_K)

        # Load K and V tiles
        k_offsets = head_offset + (k_start + k_range[:, None]) * HEAD_DIM + d_range[None, :]
        k_mask = (k_start + k_range[:, None]) < seq_len
        k_block = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)

        # Compute attention scores in tiles
        scores = tl.dot(q_block, tl.trans(k_block)) * scale

        # Online softmax (numerically stable)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_scores, block_max)

        # Rescale previous accumulated values
        decay = tl.exp(max_scores - new_max)
        output_acc = output_acc * decay[:, None]

        # Compute new softmax values
        exp_scores = tl.exp(scores - new_max[:, None])
        sum_exp = sum_exp * decay + tl.sum(exp_scores, axis=1)
        max_scores = new_max

        # Load V block and accumulate
        v_offsets = head_offset + (k_start + k_range[:, None]) * HEAD_DIM + d_range[None, :]
        v_mask = (k_start + k_range[:, None]) < seq_len
        v_block = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)

        # Accumulate: exp_scores @ V
        output_acc += tl.dot(exp_scores, v_block)

    # Final normalization
    output = output_acc / sum_exp[:, None]

    # Store result
    out_offsets = head_offset + (q_start + q_range[:, None]) * HEAD_DIM + d_range[None, :]
    out_mask = (q_start + q_range[:, None]) < seq_len
    tl.store(output_ptr + out_offsets, output, mask=out_mask)
```

**Flash Attention Benefits:**
```python
FLASH_ATTENTION_BENEFITS = {
    'memory_efficiency': {
        'complexity': 'O(N) vs O(N²) for standard attention',
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

### 3. MSA Row Attention with Pair Bias

#### Mathematical Operation

MSA Row Attention computes attention across residues within each MSA sequence, biased by the pair representation:

$$\begin{aligned}
Q, K, V &= W_Q \cdot \text{MSA}, W_K \cdot \text{MSA}, W_V \cdot \text{MSA} \\
b &= W_b \cdot \text{Pair} \quad \text{(pair bias)} \\
\text{Attention} &= \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + b\right) V
\end{aligned}$$

**Implementation Strategy:**
1. Use PyTorch Linear layers for Q, K, V projections (compute-bound, already optimal)
2. Use Triton Flash Attention kernel for attention computation (memory-bound)
3. Integrate pair bias after attention (simplified version)

**Full optimization** would integrate pair bias directly into the Flash Attention kernel for maximum efficiency.

### 4. Triangle Multiplicative Updates

#### Mathematical Operation

Triangle updates implement geometric reasoning in the pair representation:

**Outgoing:**
$$z_{ij} = \sum_k \text{gate}(p_{ik}) \odot W_{\text{left}} \cdot p_{ik} \times \text{gate}(p_{jk}) \odot W_{\text{right}} \cdot p_{jk}$$

**Incoming:**
$$z_{ij} = \sum_k \text{gate}(p_{ki}) \odot W_{\text{left}} \cdot p_{ki} \times \text{gate}(p_{kj}) \odot W_{\text{right}} \cdot p_{kj}$$

**Optimization Strategy:**

In Version 3, we use:
- **Triton LayerNorm** for input normalization (fused kernel)
- **PyTorch Linear layers** for gate/projection operations (compute-bound, optimal with rocBLAS)
- **PyTorch einsum** for triangle multiplication (already highly optimized)

The key optimization is **kernel fusion** through fused LayerNorm, reducing memory bandwidth requirements.

### 5. Outer Product Mean

#### Mathematical Operation

Projects MSA features onto the pair representation:

$$\text{Pair}_{ij} = \frac{1}{N_{\text{seqs}}} \sum_n (W \cdot \text{MSA}_n)_i \otimes (W \cdot \text{MSA}_n)_j$$

**Optimization:**
- Triton LayerNorm for MSA normalization
- PyTorch Linear for projection to outer product dimension
- PyTorch einsum for outer product computation (already optimal)
- PyTorch Linear for projection to pair dimension

## Hybrid Optimization Strategy

Version 3 employs a **hybrid optimization approach**:

### Memory-Bound Operations → Triton Kernels
- **LayerNorm**: Fused statistics computation and normalization
- **Attention**: Flash Attention with tiled computation
- **Element-wise operations**: Fused when beneficial

### Compute-Bound Operations → PyTorch/rocBLAS
- **Matrix multiplication (GEMM)**: rocBLAS is already optimal
- **Linear layers**: Highly optimized in PyTorch
- **Einsum operations**: PyTorch implementation is efficient

**Why This Approach?**

Custom Triton kernels for GEMM operations would be:
- **8-10x slower** than rocBLAS on AMD GPUs
- More complex to implement and maintain
- No performance benefit

By using Triton **only for memory-bound operations**, we achieve:
- Maximum performance gains where it matters
- Simpler implementation and maintenance
- Best of both worlds: custom kernels + optimized libraries

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
python3 tiny_openfold_v3.py
```

**Expected Output:**
```
=== TINY OPENFOLD - VERSION 3: TRITON CUSTOM KERNELS ===
Model V3 Configuration:
   MSA dimension: 64
   Pair dimension: 128
   Evoformer blocks: 4
   Total parameters: 2,641,728
   Model size: 10.6 MB (FP32)

Triton Kernel Optimizations:
   layernorm: ACTIVE
   flash_attention_msa_row: ACTIVE
   flash_attention_msa_col: ACTIVE
   flash_attention_triangle: ACTIVE

Performance Summary V3:
   Average training speed: 150-200 samples/sec
   Peak memory usage: 80-100 MB
```

### 3. Compare with Baseline

Run performance comparison:

```bash
# Compare V1, V2, V3
./launch_performance_study.sh
```

### 4. Profile Performance

Run comprehensive profiling:

```bash
# Triton-specific profiling
python3 run_triton_profiling.py
```

## Performance Analysis

### Expected Performance Gains

| Component | Baseline Time | Version 2 Time | Version 3 Time | V3 Speedup | V3 vs V2 |
|-----------|---------------|----------------|----------------|------------|----------|
| LayerNorm | 100% | 65-75% | 40-50% | 2.0-2.5x | 1.3-1.6x |
| MSA Attention | 100% | 60-80% | 35-50% | 2.0-2.9x | 1.4-2.0x |
| Triangle Attention | 100% | 60-80% | 35-50% | 2.0-2.9x | 1.4-2.0x |
| **Overall** | **100%** | **60-75%** | **35-50%** | **2.0-2.9x** | **1.3-1.7x** |

### Memory Efficiency

| Metric | Standard PyTorch | Version 2 Fused | Version 3 Triton | Improvement |
|--------|------------------|-----------------|------------------|-------------|
| Peak Memory | 196 MB | 120-140 MB | 80-100 MB | 50-60% reduction |
| Memory Bandwidth | 100% | 65-75% | 40-55% | 45-60% reduction |
| Kernel Launches | 100% | 40-60% | 20-35% | 65-80% reduction |

## Advanced Topics

### Kernel Optimization Strategies

1. **Block Size Tuning**
   - Match hardware characteristics (MI300X: 32-128 typical)
   - Optimize for occupancy (threads per SM)
   - Consider memory coalescing requirements

2. **Memory Access Patterns**
   - Minimize global memory access
   - Maximize register usage
   - Optimize cache utilization
   - Ensure coalesced memory access

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

## Troubleshooting

### Common Issues

1. **Triton Not Found**
   ```bash
   pip install triton
   # Or check environment setup
   ```

2. **Kernel Compilation Errors**
   - Verify GPU compatibility (AMD MI300X)
   - Check ROCm installation
   - Review tensor dimensions

3. **Performance Regression**
   - Ensure proper warmup (Triton JIT compilation)
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
   triton_layernorm = TritonLayerNorm(dim)
   # Benchmark just this component
   ```

2. **Compare Block Sizes**
   ```python
   # Test different configurations
   for block_size in [32, 64, 128, 256]:
       # Measure performance
   ```

3. **Memory Pattern Analysis**
   ```python
   # Check memory access efficiency
   torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
   ```

## Integration with ROCm Tools

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

## Next Steps

After completing Version 3:

1. **Review Performance Gains**: Compare with V1 and V2
2. **Understand Optimization Principles**: Kernel design patterns
3. **Experiment with Configurations**: Different block sizes and strategies

## Resources

### Documentation
- [Triton Language Tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [GPU Architecture Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/Programming-Guides.html)
- [ROCm Profiler Documentation](https://rocmdocs.amd.com/en/latest/ROCm_Tools/ROCm-Tools.html)

### Papers and References
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [AlphaFold 2 Paper](https://www.nature.com/articles/s41586-021-03819-2)
- [OpenFold Implementation](https://github.com/aqlaboratory/openfold)
- [Triton: A Language for AI Kernel Programming](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

### AMD ROCm Resources
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- [Performance Optimization Tips](https://rocmdocs.amd.com/en/latest/Programming_Guides/Opencl-programming-guide.html)

## Summary

Version 3 demonstrates the power of custom Triton kernels for optimizing memory-bound operations in the Evoformer architecture. By combining Triton kernels for memory-intensive operations with PyTorch's optimized libraries for compute-bound operations, we achieve significant performance improvements while maintaining code clarity and correctness.

**Key Takeaways:**
1. Triton enables high-level GPU kernel programming
2. Hybrid optimization (Triton + PyTorch) is often optimal
3. Memory-bound operations benefit most from custom kernels
4. Flash Attention provides significant memory and speed improvements
5. Proper kernel fusion reduces memory bandwidth requirements

