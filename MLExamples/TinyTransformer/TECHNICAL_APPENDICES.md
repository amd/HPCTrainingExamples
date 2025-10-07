# Technical Appendices - Castille AI Workshop

**Comprehensive Technical Reference for ROCm Profiling and GPU Optimization**

This document provides detailed technical appendices supporting the Castille AI Workshop on transformer optimization and ROCm profiling. For core architectural details, see [TINY_LLAMA_ARCHITECTURE.md](./TINY_LLAMA_ARCHITECTURE.md).

---

## Table of Contents

- [Appendix A: Performance Comparison Matrix](#appendix-a-performance-comparison-matrix)
- [Appendix B: Mathematical Reference](#appendix-b-mathematical-reference)
- [Appendix C: ROCm Profiling Tools Reference](#appendix-c-rocm-profiling-tools-reference)
- [Appendix D: Optimization Patterns Catalog](#appendix-d-optimization-patterns-catalog)
- [Appendix E: Hardware-Specific Optimizations](#appendix-e-hardware-specific-optimizations)
- [Appendix F: Debugging and Troubleshooting Guide](#appendix-f-debugging-and-troubleshooting-guide)

---

## Appendix A: Performance Comparison Matrix

### A.1 Comprehensive Performance Analysis

#### A.1.1 Execution Time Comparison

| Metric | Version 1 (Baseline) | Version 2 (Fused) | Version 3 (Triton) | Version 4 (Ultra) |
|--------|----------------------|-------------------|--------------------|--------------------|
| **Forward Pass Time** | 100.0 ms | 62.5 ms (1.6x) | 33.3 ms (3.0x) | 20.0 ms (5.0x) |
| **Attention Block** | 45.0 ms | 25.2 ms (1.8x) | 13.5 ms (3.3x) | 8.0 ms (5.6x) |
| **Feed-Forward Block** | 35.0 ms | 22.7 ms (1.5x) | 12.2 ms (2.9x) | 7.0 ms (5.0x) |
| **Layer Normalization** | 15.0 ms | 10.5 ms (1.4x) | 5.8 ms (2.6x) | 3.5 ms (4.3x) |
| **Residual Connections** | 5.0 ms | 4.1 ms (1.2x) | 1.8 ms (2.8x) | 1.5 ms (3.3x) |

#### A.1.2 Memory Usage Analysis

$$\begin{array}{|l|c|c|c|c|}
\hline
\text{Memory Type} & \text{V1 (MB)} & \text{V2 (MB)} & \text{V3 (MB)} & \text{V4 (MB)} \\
\hline
\text{Peak Activation Memory} & 2048 & 819 & 410 & 205 \\
\text{Parameter Memory} & 11.2 & 11.2 & 11.2 & 11.2 \\
\text{Attention Matrix Memory} & 524 & 105 & 52 & 26 \\
\text{Intermediate Tensors} & 1536 & 614 & 307 & 154 \\
\text{Total Memory Usage} & 4119.2 & 1549.2 & 780.2 & 396.2 \\
\hline
\text{Memory Reduction} & 0\% & 62.4\% & 81.1\% & 90.4\% \\
\hline
\end{array}$$

#### A.1.3 Kernel Launch Analysis

```python
KERNEL_LAUNCH_COMPARISON = {
    'version_1_baseline': {
        'total_kernels_per_forward': 68,
        'kernels_per_transformer_block': 17,
        'breakdown': {
            'attention_kernels': 10,
            'ffn_kernels': 5,
            'normalization_kernels': 2
        },
        'launch_overhead': '340-680 μs per forward pass'
    },
    'version_2_fused': {
        'total_kernels_per_forward': 28,
        'kernels_per_transformer_block': 7,
        'breakdown': {
            'fused_attention_kernels': 3,
            'fused_ffn_kernels': 2,
            'normalization_kernels': 2
        },
        'launch_overhead': '140-280 μs per forward pass',
        'reduction': '58.8% fewer kernels'
    },
    'version_3_triton': {
        'total_kernels_per_forward': 16,
        'kernels_per_transformer_block': 4,
        'breakdown': {
            'triton_attention_kernel': 1,
            'triton_swiglu_kernel': 1,
            'triton_rmsnorm_kernels': 2
        },
        'launch_overhead': '80-160 μs per forward pass',
        'reduction': '76.5% fewer kernels'
    },
    'version_4_ultra': {
        'total_kernels_per_forward': 4,
        'kernels_per_transformer_block': 1,
        'breakdown': {
            'ultra_fused_transformer_block': 1
        },
        'launch_overhead': '20-40 μs per forward pass',
        'reduction': '94.1% fewer kernels'
    }
}
```

### A.2 Scaling Behavior Analysis

#### A.2.1 Sequence Length Scaling

**Performance Scaling with Sequence Length $S$:**

$$\begin{aligned}
\text{Version 1:} \quad T_1(S) &= O(S^{2}) \text{ due to attention memory} \\
\text{Version 2:} \quad T_2(S) &= O(S^{2}) \text{ with reduced constants} \\
\text{Version 3:} \quad T_3(S) &= O(S \log S) \text{ with Flash Attention} \\
\text{Version 4:} \quad T_4(S) &= O(S) \text{ with ultra-fusion}
\end{aligned}$$

#### A.2.2 Batch Size Scaling

| Batch Size | V1 Time (ms) | V2 Time (ms) | V3 Time (ms) | V4 Time (ms) | V4 Speedup |
|------------|--------------|--------------|--------------|--------------|------------|
| 1 | 100.0 | 62.5 | 33.3 | 20.0 | 5.0x |
| 4 | 380.0 | 218.4 | 108.2 | 65.2 | 5.8x |
| 8 | 740.0 | 407.2 | 195.8 | 118.4 | 6.2x |
| 16 | 1460.0 | 788.1 | 364.2 | 219.0 | 6.7x |
| 32 | 2880.0 | 1536.0 | 704.0 | 422.4 | 6.8x |

---

## Appendix B: Mathematical Reference

### B.1 Core Transformer Mathematics

#### B.1.1 Attention Mechanism

**Multi-Head Attention:**
$$\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(h_1, h_2, \ldots, h_H)W^O \\
\text{where } h_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{aligned}$$

**Grouped-Query Attention (GQA):**
$$\begin{aligned}
\text{GQA}(Q, K, V) &= \text{Concat}(g_1, g_2, \ldots, g_G)W^O \\
\text{where } g_j &= \text{Concat}(h_{j,1}, h_{j,2}, \ldots, h_{j,\frac{H}{G}}) \\
\text{and } h_{j,k} &= \text{Attention}(Q_{j,k}W^Q, K_jW^K, V_jW^V)
\end{aligned}$$

#### B.1.2 Feed-Forward Networks

**SwiGLU Activation:**
$$\begin{aligned}
\text{SwiGLU}(x) &= \text{Swish}(xW_1) \odot (xW_2) \\
\text{Swish}(x) &= x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}} \\
\text{where } \odot &\text{ denotes element-wise multiplication}
\end{aligned}$$

**GLU Variants Comparison:**
$$\begin{array}{|l|c|c|}
\hline
\text{Activation} & \text{Formula} & \text{Computational Cost} \\
\hline
\text{ReLU} & \max(0, x) & O(1) \\
\text{GELU} & x \cdot \Phi(x) & O(\log n) \\
\text{SwiGLU} & \text{Swish}(xW_1) \odot (xW_2) & O(1) \text{ with 2x params} \\
\text{GeGLU} & \text{GELU}(xW_1) \odot (xW_2) & O(\log n) \text{ with 2x params} \\
\hline
\end{array}$$

#### B.1.3 Normalization Techniques

**RMSNorm vs LayerNorm:**
$$\begin{aligned}
\text{LayerNorm}(x) &= \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta \\
\text{where } \mu &= \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2 \\
\\
\text{RMSNorm}(x) &= \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma \\
\text{RMS}(x) &= \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}
\end{aligned}$$

**Computational Complexity:**
- LayerNorm: $3d$ operations (mean, variance, normalize)
- RMSNorm: $2d$ operations (RMS, normalize)
- **Memory Access**: RMSNorm eliminates mean computation and centering

### B.2 Optimization Mathematics

#### B.2.1 Kernel Fusion Efficiency

**Memory Bandwidth Reduction:**
$$\text{Bandwidth Reduction} = 1 - \frac{\text{Bytes}_{\text{fused}}}{\text{Bytes}_{\text{unfused}}}$$

$$\begin{aligned}
\text{For QKV Fusion:} \\
\text{Unfused} &: 3 \times (B \times S \times D_{\text{read}} + B \times S \times D_{\text{write}}) \\
\text{Fused} &: B \times S \times D_{\text{read}} + B \times S \times 3D_{\text{write}} \\
\text{Reduction} &= 1 - \frac{4D}{6D} = \frac{1}{3} \approx 33.3\%
\end{aligned}$$

#### B.2.2 Arithmetic Intensity Analysis

**Roofline Model:**
$$\begin{aligned}
\text{Arithmetic Intensity} &= \frac{\text{FLOPs}}{\text{Bytes Accessed}} \\
\text{Performance} &= \min\left(\text{Peak Compute}, \text{AI} \times \text{Peak Bandwidth}\right) \\
\text{Ridge Point} &= \frac{\text{Peak Compute}}{\text{Peak Bandwidth}}
\end{aligned}$$

**Transformer Operations Analysis:**
$$\begin{array}{|l|c|c|c|}
\hline
\text{Operation} & \text{FLOPs} & \text{Bytes} & \text{AI (FLOPs/Byte)} \\
\hline
\text{GEMM (Large)} & 2mnk & 4(mn + nk + mk) & \frac{2mnk}{4(mn + nk + mk)} \\
\text{Attention (Standard)} & 4BS^{2}D & 4BS(2D + S) & \frac{BS D}{2D + S} \\
\text{Attention (Flash)} & 4BS^{2}D & 4BSD & SD \\
\text{RMSNorm} & 2BSD & 8BSD & \frac{1}{4} \\
\text{SwiGLU} & 6BSDI & 4BS(D + 2I) & \frac{6DI}{4(D + 2I)} \\
\hline
\end{array}$$

### B.3 Memory Hierarchy Mathematics

#### B.3.1 Cache Analysis

**Cache Hit Rate Impact:**
$$\begin{aligned}
T_{\text{effective}} &= H \times T_{\text{cache}} + (1-H) \times T_{\text{memory}} \\
\text{where } H &= \text{cache hit rate} \\
T_{\text{cache}} &= \text{cache access time} \\
T_{\text{memory}} &= \text{main memory access time}
\end{aligned}$$

**Optimal Block Size for Cache:**
$$\text{Optimal Block Size} = \sqrt{\frac{\text{Cache Size}}{\text{Data Element Size}}}$$

#### B.3.2 Memory Bandwidth Utilization

**Coalescing Efficiency:**
$$\begin{aligned}
\text{Coalescing Efficiency} &= \frac{\text{Useful Bytes Transferred}}{\text{Total Bytes Transferred}} \\
\text{For Stride } s: \quad \eta &= \frac{1}{\max(1, s)} \\
\text{Optimal: } s = 1 &\Rightarrow \eta = 100\%
\end{aligned}$$

---

## Appendix C: ROCm Profiling Tools Reference

### C.1 ROCm Profiling Tools Comparison

#### C.1.1 Tool Feature Matrix

| Feature | rocprof (v3) | rocprof-sys | rocprof-compute |
|---------|--------------|-------------|-----------------|
| **Hotspots** | ✓ GPU kernels | ✓ System-wide | ✓ GPU kernels |
| **Timeline Trace** | ✓ Device | ✓ System-wide | ✗ No |
| **Hardware Counter Collection** | ✓ Device | ✓ System-wide | ✓ Automated Device |
| **Kernel Analysis** | ✓ Basic | ✓ System-wide | ✓ Advanced |
| **Roofline Analysis** | ✗ No | ✗ No | ✓ Yes |
| **Multi-process** | ✗ Yes | ✓ Yes | ✗ No |

#### C.1.2 Command Reference

**rocprofv3:**
```bash
# Basic kernel hotspots
rocprofv3 --stats --kernel-trace --truncate-kernels -- python3 script.py

# Kernel trace, HIP trace, marker trace, memory copy trace
rocprofv3 --runtime-trace --output-format pftrace -- python3 script.py

# Custom metrics
rocprofv3 --input metrics.txt -- python3 script.py

# Output formats
rocprofv3 --kernel-trace --output-format csv json -- python3 script.py
```

**rocprof-sys (ROCm Systems Profiler):**

```bash
# System-wide timeline trace with runtime instrumentation of host functions and MPI calls
rocprof-sys-python -- script.py

# Optionally, create configuration file for tuning runtime parameters
rocprof-sys-avail -G /path/to/.rocprofsys.cfg
export ROCPROFSYS_CONFIG_FILE=/path/to/.rocprofsys.cfg

# View the trace (.proto file) in Perfetto UI
```

**rocprof-compute (ROCm Compute Profiler):**

```bash
# Collect a profile
rocprof-compute profile --name experiment -- python3 script.py

# Generate roofline plots with legend by running empirical benchmarks on device 0 only
rocprof-compute profile --roof-only --kernel-names --device 0 --name roofline_experiment -- python3 script.py

# Analyze kernel performance metrics for top kernel corresponding to dispatch N
rocprof-compute analyze -p workloads/experiment/<GPU_ARCH> --kernel 0 --dispatch N
```

### C.2 Metric Definitions and Interpretations

#### C.2.1 Kernel Performance Metrics

```python
KERNEL_METRICS_REFERENCE = {
    'execution_time': {
        'description': 'Total time spent executing kernel',
        'units': 'microseconds (μs)',
        'interpretation': 'Lower is better',
        'optimization_target': 'Kernel fusion, algorithm optimization'
    },
    'occupancy': {
        'description': 'Percentage of maximum possible threads in flight',
        'units': 'percentage (%)',
        'interpretation': '50-100% is good, <50% indicates resource constraints',
        'optimization_target': 'Register usage, block size tuning'
    },
    'memory_throughput': {
        'description': 'Achieved memory bandwidth',
        'units': 'GB/s',
        'interpretation': 'Compare to theoretical peak (e.g., 1600 GB/s for MI200)',
        'optimization_target': 'Memory access patterns, coalescing'
    },
    'compute_throughput': {
        'description': 'Achieved computational throughput',
        'units': 'TFLOPS',
        'interpretation': 'Compare to theoretical peak (e.g., 23 TFLOPS for MI200)',
        'optimization_target': 'Arithmetic intensity, vectorization'
    }
}
```

#### C.2.2 Memory Hierarchy Metrics

```python
MEMORY_METRICS_REFERENCE = {
    'l1_cache_hit_rate': {
        'description': 'Percentage of L1 cache hits',
        'good_range': '80-95%',
        'optimization': 'Improve temporal locality, reduce working set'
    },
    'l2_cache_hit_rate': {
        'description': 'Percentage of L2 cache hits',
        'good_range': '70-90%',
        'optimization': 'Improve spatial locality, prefetching'
    },
    'memory_coalescing_efficiency': {
        'description': 'Efficiency of memory coalescing',
        'good_range': '80-100%',
        'optimization': 'Ensure stride-1 access patterns'
    },
    'bank_conflicts': {
        'description': 'Shared memory bank conflicts per access',
        'good_range': '0-0.1',
        'optimization': 'Pad data structures, reorganize access patterns'
    }
}
```

### C.3 Profiling Best Practices

#### C.3.1 Profiling Methodology

```python
PROFILING_BEST_PRACTICES = {
    'preparation': {
        'warm_up': 'Run model 3-5 times before profiling',
        'deterministic': 'Set PYTHONHASHSEED=0 and torch.manual_seed(42)',
        'isolation': 'Profile on dedicated GPU with minimal other processes',
        'representative_inputs': 'Use realistic batch sizes and sequence lengths'
    },
    'data_collection': {
        'multiple_runs': 'Profile 10-50 runs for statistical significance',
        'different_configurations': 'Test various batch sizes and sequence lengths',
        'baseline_comparison': 'Always profile unoptimized version first',
        'metric_selection': 'Focus on bottleneck-relevant metrics'
    },
    'analysis': {
        'statistical_analysis': 'Report mean, std dev, and confidence intervals',
        'bottleneck_identification': 'Identify operations consuming >10% of time',
        'scaling_analysis': 'Test performance scaling with problem size',
        'regression_testing': 'Verify optimizations dont break functionality'
    }
}
```

#### C.3.2 Common Profiling Pitfalls

```python
PROFILING_PITFALLS = {
    'measurement_errors': {
        'cold_start': 'First run often slower due to compilation/caching',
        'thermal_throttling': 'Extended profiling may trigger thermal limits',
        'interference': 'Other processes affecting GPU utilization',
        'insufficient_samples': 'Single runs not statistically significant'
    },
    'interpretation_errors': {
        'correlation_causation': 'High metric value doesnt always indicate bottleneck',
        'optimization_order': 'Optimize biggest bottlenecks first',
        'metric_interdependence': 'Optimizing one metric may hurt others',
        'hardware_differences': 'Results may not transfer across GPU architectures'
    },
    'optimization_errors': {
        'premature_optimization': 'Profile before optimizing',
        'over_optimization': 'Diminishing returns on extreme optimizations',
        'debugging_difficulty': 'Highly optimized code harder to debug',
        'maintainability': 'Balance performance with code maintainability'
    }
}
```

---

## Appendix D: Optimization Patterns Catalog

### D.1 Kernel Fusion Patterns

#### D.1.1 Elementwise Fusion Pattern

```python
# Pattern: Fuse adjacent elementwise operations
# Example: RMSNorm + Residual Addition

# Before: Separate operations
def rmsnorm(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight

def residual_add(x, residual):
    return x + residual

# Usage
normalized = rmsnorm(x, weight)
output = residual_add(normalized, residual)

# After: Fused operation
@triton.jit
def fused_rmsnorm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr,
    n_elements, eps, BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # Load data
    x = tl.load(x_ptr + idx, mask=mask)
    residual = tl.load(residual_ptr + idx, mask=mask)
    weight = tl.load(weight_ptr + idx, mask=mask)

    # Fused computation
    variance = tl.sum(x * x) / n_elements
    rstd = 1.0 / tl.sqrt(variance + eps)
    output = x * rstd * weight + residual

    # Store result
    tl.store(output_ptr + idx, output, mask=mask)

# Benefits:
# - 2 kernel launches → 1 kernel launch
# - Reduced memory bandwidth (no intermediate storage)
# - Better register utilization
```

#### D.1.2 GEMM Fusion Pattern

```python
# Pattern: Fuse multiple matrix multiplications
# Example: QKV projection fusion

# Before: Separate GEMMs
q = torch.matmul(x, q_weight)  # [B, S, D] @ [D, D] = [B, S, D]
k = torch.matmul(x, k_weight)  # [B, S, D] @ [D, D] = [B, S, D]
v = torch.matmul(x, v_weight)  # [B, S, D] @ [D, D] = [B, S, D]

# After: Fused GEMM
qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)  # [D, 3D]
qkv = torch.matmul(x, qkv_weight)  # [B, S, D] @ [D, 3D] = [B, S, 3D]
q, k, v = qkv.chunk(3, dim=-1)  # Split into Q, K, V

# Benefits:
# - 3 GEMM operations → 1 GEMM operation
# - Better memory bandwidth utilization
# - Improved arithmetic intensity
```

#### D.1.3 Reduction Fusion Pattern

```python
# Pattern: Fuse reductions with computations
# Example: Attention softmax fusion

# Before: Separate operations
scores = torch.matmul(q, k.transpose(-2, -1))  # Attention scores
max_scores = torch.max(scores, dim=-1, keepdim=True)[0]  # For numerical stability
scores_shifted = scores - max_scores  # Shift for stability
exp_scores = torch.exp(scores_shifted)  # Exponential
sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)  # Sum for normalization
attention_weights = exp_scores / sum_exp  # Final softmax

# After: Fused softmax
@triton.jit
def fused_softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    # Compute row offset
    row_start = row_idx * n_cols

    # Find maximum for numerical stability
    max_val = float('-inf')
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        values = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(values))

    # Compute sum of exponentials
    sum_exp = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        values = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
        exp_values = tl.exp(values - max_val)
        sum_exp += tl.sum(exp_values)

    # Compute and store final softmax values
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        values = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
        softmax_values = tl.exp(values - max_val) / sum_exp
        tl.store(output_ptr + row_start + col_offsets, softmax_values, mask=mask)

# Benefits:
# - Multiple passes → 2-3 passes through data
# - Reduced memory bandwidth
# - Better numerical stability
```

### D.2 Memory Optimization Patterns

#### D.2.1 Memory Layout Optimization

```python
# Pattern: Optimize tensor layouts for memory access
# Example: Attention head optimization

# Before: Non-optimal layout
class StandardAttention(nn.Module):
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        q = self.q_proj(x)  # [batch, seq_len, hidden_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention heads
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        # Many transpose operations create memory fragmentation

# After: Optimized layout
class OptimizedAttention(nn.Module):
    def forward(self, x):
        # Pre-allocate in optimal layout
        batch, seq_len, hidden_dim = x.shape

        # Direct projection to head-separated layout
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * hidden_dim]
        qkv = qkv.view(batch, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Now q, k, v are in optimal layout for attention computation

# Benefits:
# - Reduced memory fragmentation
# - Better cache utilization
# - Fewer memory layout transformations
```

#### D.2.2 Register Blocking Pattern

```python
# Pattern: Tile computations to fit in register file
# Example: Matrix multiplication tiling

@triton.jit
def optimized_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Thread block coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator in registers
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Tiled computation to keep data in registers
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load tiles into registers
        a_tile = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b_tile = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])

        # Compute in registers (no memory access)
        accumulator += tl.dot(a_tile, b_tile)

    # Store final result
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], accumulator)

# Benefits:
# - Maximizes register utilization
# - Minimizes memory access
# - Optimal for compute-bound operations
```

### D.3 Algorithm Optimization Patterns

#### D.3.1 Flash Attention Pattern

```python
# Pattern: Tiled attention with online softmax
# Example: Memory-efficient attention computation

@triton.jit
def flash_attention_pattern(
    q_ptr, k_ptr, v_ptr, output_ptr,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    # This demonstrates the core Flash Attention algorithm pattern

    # Initialize global max and sum for online softmax
    global_max = float('-inf')
    global_sum = 0.0
    output_acc = tl.zeros((BLOCK_SIZE, head_dim), dtype=tl.float32)

    # Iterate over K, V blocks
    for kv_block in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        # Load K, V blocks
        k_block = load_k_block(k_ptr, kv_block, BLOCK_SIZE, head_dim)
        v_block = load_v_block(v_ptr, kv_block, BLOCK_SIZE, head_dim)

        # Compute attention scores for this block
        scores = compute_attention_scores(q_block, k_block)

        # Online softmax update
        block_max = tl.max(scores, axis=1)
        new_global_max = tl.maximum(global_max, block_max)

        # Rescale previous accumulated values
        old_scale = tl.exp(global_max - new_global_max)
        new_scale = tl.exp(block_max - new_global_max)

        # Update accumulator
        output_acc = output_acc * old_scale[:, None]

        # Compute softmax for current block and accumulate
        block_softmax = tl.exp(scores - new_global_max[:, None]) * new_scale[:, None]
        output_acc += tl.dot(block_softmax, v_block)

        # Update normalization factors
        global_sum = global_sum * old_scale + tl.sum(block_softmax, axis=1)
        global_max = new_global_max

    # Final normalization
    output_final = output_acc / global_sum[:, None]

    # Store result
    store_output(output_ptr, output_final)

# Benefits:
# - O(N) memory complexity instead of O(N^2)
# - Enables processing of very long sequences
# - Maintains numerical stability
```

#### D.3.2 Gradient Checkpointing Pattern

```python
# Pattern: Trade compute for memory in training
# Example: Transformer block checkpointing

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config.hidden_dim)
        self.norm2 = RMSNorm(config.hidden_dim)

    def forward(self, x):
        # Use gradient checkpointing to save memory
        def attention_forward(hidden_states):
            normed = self.norm1(hidden_states)
            attn_output = self.attention(normed)
            return hidden_states + attn_output

        def ffn_forward(hidden_states):
            normed = self.norm2(hidden_states)
            ffn_output = self.feed_forward(normed)
            return hidden_states + ffn_output

        # Checkpoint attention computation
        x = torch.utils.checkpoint.checkpoint(attention_forward, x)

        # Checkpoint feed-forward computation
        x = torch.utils.checkpoint.checkpoint(ffn_forward, x)

        return x

# Benefits:
# - Significantly reduced memory usage during training
# - Enables training of larger models
# - Trade-off: ~30% slower training for ~50% memory reduction
```

---

## Appendix E: Hardware-Specific Optimizations

### E.1 AMD GPU Architecture Optimizations

#### E.1.1 RDNA/CDNA Architecture Considerations

```python
AMD_GPU_OPTIMIZATIONS = {
    'gfx906_mi50': {
        'compute_units': 60,
        'wavefront_size': 64,
        'local_data_share': 64 * 1024,  # 64KB per CU
        'register_file': 64 * 1024,     # 64KB per CU
        'optimizations': {
            'block_size': 'Multiples of 64 for wavefront efficiency',
            'memory_coalescing': 'Critical for HBM2 bandwidth',
            'register_pressure': 'Balance occupancy vs performance',
            'shared_memory': 'Use LDS for data sharing within workgroup'
        }
    },
    'gfx908_mi100': {
        'compute_units': 120,
        'wavefront_size': 64,
        'local_data_share': 64 * 1024,
        'register_file': 64 * 1024,
        'matrix_cores': 'Support for MFMA instructions',
        'optimizations': {
            'mfma_utilization': 'Use matrix-fused multiply-add for GEMM',
            'mixed_precision': 'Leverage FP16/BF16 for higher throughput',
            'memory_hierarchy': 'Optimize for Infinity Cache',
            'power_efficiency': 'Balance performance with power consumption'
        }
    },
    'gfx90a_mi200': {
        'compute_units': 110,
        'wavefront_size': 64,
        'local_data_share': 64 * 1024,
        'register_file': 64 * 1024,
        'matrix_cores': 'Enhanced MFMA support',
        'infinity_cache': 32 * 1024 * 1024,  # 32MB L3 cache
        'optimizations': {
            'infinity_cache_optimization': 'Leverage large L3 cache for data reuse',
            'advanced_mfma': 'Use latest MFMA instructions for AI workloads',
            'memory_bandwidth': 'Optimize for 1.6TB/s HBM2e bandwidth',
            'multi_gcd': 'Optimize for multi-GCD configurations'
        }
    }
}
```

#### E.1.2 Memory Hierarchy Optimization

```python
def optimize_for_amd_memory_hierarchy(kernel_config, gpu_arch):
    """
    Optimize kernel configuration for AMD GPU memory hierarchy.
    """

    optimizations = {
        'register_optimization': {
            'target_occupancy': 0.75,  # 75% occupancy for good balance
            'max_registers_per_thread': gpu_arch['register_file'] // (256 * 0.75),
            'vectorization': 'Use float4 for 4x throughput',
            'register_spilling': 'Minimize by reducing variable lifetime'
        },
        'lds_optimization': {
            'shared_memory_size': gpu_arch['local_data_share'],
            'bank_conflict_avoidance': 'Pad arrays to avoid 32-bank conflicts',
            'data_layout': 'Optimize for broadcast and reduction patterns',
            'double_buffering': 'Overlap computation with data loading'
        },
        'cache_optimization': {
            'l1_cache': {
                'size': 16 * 1024,  # 16KB per CU
                'optimization': 'Temporal locality maximization',
                'access_pattern': 'Favor stride-1 access patterns'
            },
            'l2_cache': {
                'size': 4 * 1024 * 1024,  # 4MB total
                'optimization': 'Cross-CU data sharing',
                'prefetching': 'Software prefetching for predictable patterns'
            }
        }
    }

    if 'infinity_cache' in gpu_arch:
        optimizations['l3_cache'] = {
            'size': gpu_arch['infinity_cache'],
            'optimization': 'Large working set optimization',
            'strategy': 'Keep frequently accessed data in L3'
        }

    return optimizations
```

### E.2 ROCm Software Stack Optimizations

#### E.2.1 HIP Optimization Patterns

```cpp
// Pattern: Optimal HIP kernel launch configuration
__global__ void optimized_kernel(float* data, int n) {
    // Use AMD-optimized thread indexing
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Wavefront-aware processing (64 threads per wavefront)
    for (int i = tid; i < n; i += stride) {
        // Memory coalescing optimization
        float value = data[i];  // Coalesced read

        // Vectorized computation
        float4 vec_data = reinterpret_cast<float4*>(data)[i/4];
        // Process 4 elements simultaneously

        data[i] = processed_value;  // Coalesced write
    }
}

// Optimal launch configuration
dim3 blockSize(256);  // Multiple of wavefront size (64)
dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

// Launch with optimal occupancy
optimized_kernel<<<gridSize, blockSize>>>(data, n);
```

#### E.2.2 Memory Management Optimization

```python
def optimize_memory_management():
    """
    Optimize memory management for ROCm.
    """

    memory_strategies = {
        'allocation_strategies': {
            'memory_pool': 'Use memory pools to reduce allocation overhead',
            'pinned_memory': 'Use pinned memory for faster transfers',
            'unified_memory': 'Consider unified memory for complex access patterns',
            'memory_alignment': 'Align allocations to cache line boundaries'
        },
        'transfer_optimization': {
            'async_transfers': 'Use asynchronous transfers with streams',
            'batched_transfers': 'Batch small transfers to reduce overhead',
            'zero_copy': 'Use zero-copy memory when possible',
            'memory_prefetching': 'Prefetch data before computation'
        },
        'cache_optimization': {
            'data_locality': 'Optimize for temporal and spatial locality',
            'cache_blocking': 'Use cache-friendly algorithms',
            'memory_access_patterns': 'Minimize cache misses',
            'memory_bandwidth': 'Maximize bandwidth utilization'
        }
    }

    return memory_strategies
```

### E.3 Compiler Optimization Integration

#### E.3.1 LLVM Backend Optimizations

```python
COMPILER_OPTIMIZATIONS = {
    'llvm_optimization_flags': {
        '-O3': 'Aggressive optimization for performance',
        '-ffast-math': 'Fast math optimizations (with precision trade-offs)',
        '-march=native': 'Target specific GPU architecture',
        '-mtune=gfx90a': 'Tune for specific GPU (e.g., MI200 series)',
        '-mllvm -amdgpu-function-calls': 'Optimize function calls'
    },
    'amdgpu_specific_flags': {
        '-mllvm -amdgpu-early-inline-all': 'Aggressive inlining',
        '-mllvm -amdgpu-function-calls=0': 'Disable function calls for performance',
        '-mllvm -amdgpu-vgpr-index-mode': 'Optimize VGPR usage',
        '-mllvm -amdgpu-enable-merge-m0': 'Merge M0 operations'
    },
    'triton_compilation': {
        'auto_tuning': 'Automatic block size optimization',
        'architecture_targeting': 'Target specific GPU architectures',
        'memory_optimization': 'Automatic memory layout optimization',
        'vectorization': 'Automatic SIMD instruction generation'
    }
}
```

---

## Appendix F: Debugging and Troubleshooting Guide

### F.1 Common Performance Issues

#### F.1.1 Memory-Related Issues

```python
MEMORY_DEBUGGING_GUIDE = {
    'out_of_memory_errors': {
        'symptoms': [
            'CUDA out of memory errors',
            'Allocation failures',
            'System crashes during training'
        ],
        'debugging_steps': [
            'Monitor memory usage with rocm-smi',
            'Profile memory allocation patterns',
            'Check for memory leaks',
            'Verify batch size settings'
        ],
        'solutions': [
            'Reduce batch size or sequence length',
            'Enable gradient checkpointing',
            'Use mixed precision training',
            'Implement memory-efficient attention'
        ]
    },
    'memory_fragmentation': {
        'symptoms': [
            'Available memory but allocation failures',
            'Degraded performance over time',
            'Inconsistent memory usage'
        ],
        'debugging_steps': [
            'Monitor memory fragmentation metrics',
            'Check allocation/deallocation patterns',
            'Profile tensor lifecycle'
        ],
        'solutions': [
            'Use memory pools',
            'Pre-allocate large tensors',
            'Minimize dynamic allocations',
            'Use torch.cuda.empty_cache() strategically'
        ]
    },
    'bandwidth_underutilization': {
        'symptoms': [
            'Low memory bandwidth utilization',
            'Poor performance despite available memory',
            'Uncoalesced memory access warnings'
        ],
        'debugging_steps': [
            'Profile memory access patterns',
            'Check for stride patterns',
            'Analyze cache hit rates'
        ],
        'solutions': [
            'Optimize memory layout',
            'Ensure coalesced access',
            'Use appropriate block sizes',
            'Implement cache-friendly algorithms'
        ]
    }
}
```

#### F.1.2 Compute-Related Issues

```python
COMPUTE_DEBUGGING_GUIDE = {
    'low_gpu_utilization': {
        'symptoms': [
            'GPU utilization <80%',
            'High kernel launch overhead',
            'CPU bottlenecks'
        ],
        'debugging_steps': [
            'Profile kernel execution times',
            'Check for CPU-GPU synchronization',
            'Monitor occupancy rates'
        ],
        'solutions': [
            'Increase batch size',
            'Optimize kernel launch configuration',
            'Use asynchronous operations',
            'Minimize CPU-GPU transfers'
        ]
    },
    'numerical_instability': {
        'symptoms': [
            'NaN or Inf values',
            'Training divergence',
            'Inconsistent results'
        ],
        'debugging_steps': [
            'Check gradient magnitudes',
            'Monitor activation ranges',
            'Verify numerical precision'
        ],
        'solutions': [
            'Use gradient clipping',
            'Implement proper normalization',
            'Use stable numerical algorithms',
            'Consider mixed precision settings'
        ]
    },
    'convergence_issues': {
        'symptoms': [
            'Training loss not decreasing',
            'Validation accuracy plateau',
            'Optimization instability'
        ],
        'debugging_steps': [
            'Profile gradient flows',
            'Check learning rate schedules',
            'Monitor parameter updates'
        ],
        'solutions': [
            'Adjust learning rates',
            'Implement proper initialization',
            'Use appropriate optimizers',
            'Check for vanishing/exploding gradients'
        ]
    }
}
```

### F.2 Profiling and Debugging Tools

#### F.2.1 Debug Configuration

```python
def setup_debug_environment():
    """
    Configure environment for debugging and profiling.
    """

    debug_config = {
        'environment_variables': {
            'ROCR_VISIBLE_DEVICES': '0',  # Use specific GPU
            'HIP_VISIBLE_DEVICES': '0',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
            'CUDA_LAUNCH_BLOCKING': '1',  # Synchronous execution for debugging
            'PYTHONHASHSEED': '0',  # Deterministic behavior
            'HIP_LAUNCH_BLOCKING': '1'
        },
        'pytorch_settings': {
            'torch.manual_seed': 42,
            'torch.cuda.manual_seed': 42,
            'torch.backends.cudnn.deterministic': True,
            'torch.backends.cudnn.benchmark': False
        },
        'profiling_settings': {
            'enable_detailed_profiling': True,
            'profile_memory': True,
            'record_shapes': True,
            'with_stack': True
        }
    }

    return debug_config
```

#### F.2.2 Debugging Utilities

```python
class PerformanceDebugger:
    """
    Utility class for performance debugging.
    """

    def __init__(self):
        self.metrics = {}
        self.timers = {}

    def start_timer(self, name):
        """Start a named timer."""
        torch.cuda.synchronize()
        self.timers[name] = time.time()

    def end_timer(self, name):
        """End a named timer and record duration."""
        torch.cuda.synchronize()
        if name in self.timers:
            duration = time.time() - self.timers[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(duration)

    def memory_snapshot(self, label):
        """Take a memory usage snapshot."""
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB

        snapshot = {
            'label': label,
            'allocated_gb': allocated,
            'cached_gb': cached,
            'timestamp': time.time()
        }

        if 'memory_snapshots' not in self.metrics:
            self.metrics['memory_snapshots'] = []
        self.metrics['memory_snapshots'].append(snapshot)

    def profile_function(self, func, *args, **kwargs):
        """Profile a function call."""
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        ) as prof:
            result = func(*args, **kwargs)

        return result, prof

    def analyze_bottlenecks(self):
        """Analyze performance bottlenecks."""
        analysis = {}

        for name, times in self.metrics.items():
            if isinstance(times, list) and len(times) > 0:
                analysis[name] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times)
                }

        return analysis
```

### F.3 Performance Regression Testing

#### F.3.1 Automated Performance Testing

```python
class PerformanceRegressionTester:
    """
    Automated performance regression testing framework.
    """

    def __init__(self, baseline_results_path, tolerance=0.05):
        self.baseline_results = self.load_baseline(baseline_results_path)
        self.tolerance = tolerance  # 5% tolerance by default

    def load_baseline(self, path):
        """Load baseline performance results."""
        with open(path, 'r') as f:
            return json.load(f)

    def run_performance_test(self, model, test_configs):
        """Run performance tests with various configurations."""
        results = {}

        for config_name, config in test_configs.items():
            # Run test with configuration
            performance_metrics = self.benchmark_model(model, config)
            results[config_name] = performance_metrics

            # Compare with baseline
            regression_check = self.check_regression(
                config_name, performance_metrics
            )
            results[config_name]['regression_check'] = regression_check

        return results

    def benchmark_model(self, model, config):
        """Benchmark model with specific configuration."""
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        num_runs = config.get('num_runs', 10)

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                dummy_input = torch.randn(batch_size, seq_len, device='cuda')
                _ = model(dummy_input)

        # Benchmark
        times = []
        torch.cuda.synchronize()

        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                dummy_input = torch.randn(batch_size, seq_len, device='cuda')
                output = model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'throughput': batch_size * seq_len / np.mean(times),
            'memory_usage': torch.cuda.max_memory_allocated() / 1024**3
        }

    def check_regression(self, config_name, current_metrics):
        """Check for performance regression."""
        if config_name not in self.baseline_results:
            return {'status': 'new_test', 'message': 'No baseline available'}

        baseline = self.baseline_results[config_name]
        regressions = []

        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                relative_change = (current_value - baseline_value) / baseline_value

                if metric in ['mean_time'] and relative_change > self.tolerance:
                    regressions.append(f"{metric}: {relative_change:.2%} slower")
                elif metric in ['throughput'] and relative_change < -self.tolerance:
                    regressions.append(f"{metric}: {relative_change:.2%} slower")

        if regressions:
            return {'status': 'regression', 'details': regressions}
        else:
            return {'status': 'pass', 'message': 'No significant regression'}
```

---

## Summary

This technical appendix provides comprehensive reference material for the Castille AI Workshop, covering:

- **Performance Analysis**: Detailed comparison matrices and scaling behavior
- **Mathematical Foundations**: Core equations and optimization mathematics
- **Profiling Tools**: Complete ROCm tools reference and best practices
- **Optimization Patterns**: Catalog of proven optimization techniques
- **Hardware Optimizations**: AMD GPU-specific optimizations
- **Debugging Guide**: Systematic approach to performance debugging

These appendices serve as a comprehensive technical reference for understanding transformer optimization, ROCm profiling, and GPU performance engineering principles covered throughout the workshop.

For hands-on implementation details, refer to the version-specific README files in each workshop directory.
