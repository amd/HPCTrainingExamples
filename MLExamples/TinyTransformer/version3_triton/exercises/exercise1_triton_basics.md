# Exercise 1: Understanding Triton Kernel Basics

**Objective**: Learn the fundamentals of Triton GPU programming and analyze basic kernel performance.

**Time**: 45 minutes

**Prerequisites**: Completed Version 1 and Version 2 exercises

## Background

Triton is a language and compiler for writing custom GPU kernels. It provides:
- Python-like syntax for GPU programming
- Automatic memory coalescing and optimization
- Block-level programming model
- Integration with PyTorch

In this exercise, you'll analyze the basic structure of Triton kernels and understand their performance characteristics.

## Part A: Kernel Structure Analysis (15 minutes)

### Step 1: Examine the RMSNorm Kernel

Open `tiny_llama_v3.py` and locate the `rmsnorm_kernel` function:

```python
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
```

**Questions to Answer:**

1. **Pointer Management**: How does Triton handle memory pointers compared to CUDA?
2. **Block Processing**: What is the role of `BLOCK_SIZE` in this kernel?
3. **Constexpr Usage**: Why are `eps` and `BLOCK_SIZE` marked as `tl.constexpr`?
4. **Memory Access Pattern**: How does the kernel ensure coalesced memory access?

### Step 2: Analyze Memory Access Patterns

Look at the variance computation loop:

```python
for i in range(0, n_elements, BLOCK_SIZE):
    offsets = i + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_vals = tl.load(x_ptr + row_idx * n_elements + offsets, mask=mask, other=0.0)
    variance += tl.sum(x_vals * x_vals, axis=0)
```

**Analysis Tasks:**

1. **Memory Coalescing**: Explain how the `offsets` calculation ensures coalesced memory access
2. **Boundary Handling**: What does the `mask` parameter accomplish?
3. **Reduction Pattern**: How does this implement an efficient parallel reduction?

### Step 3: Compare with PyTorch Implementation

Compare the Triton RMSNorm with the PyTorch version:

```python
def pytorch_rmsnorm(x):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight
```

**Discussion Points:**

1. **Kernel Fusion**: How does Triton fuse operations that PyTorch keeps separate?
2. **Memory Efficiency**: What memory advantages does the Triton version have?
3. **Numerical Precision**: Are there any precision considerations?

## Part B: Performance Profiling (20 minutes)

### Step 4: Run Basic Profiling

Execute the Triton profiling script:

```bash
cd version3_triton/
python3 run_triton_profiling.py
```

**Expected Output Analysis:**

```
=== Triton Kernel Performance Analysis ===

1. RMSNorm Kernel Profiling
  Triton RMSNorm: X.XXX ms
  PyTorch RMSNorm: Y.YYY ms
  Speedup: Z.ZZx
  Max error: E.EEe-XX
```

**Performance Questions:**

1. **Speedup Analysis**: What speedup did you achieve? Is it consistent with expectations?
2. **Accuracy Check**: What is the maximum error between implementations? Is this acceptable?
3. **Memory Usage**: How does memory usage compare between the implementations?

### Step 5: Analyze ROCProfiler Results

Run the ROCProfiler analysis:

```bash
chmod +x run_rocprof_triton.sh
./run_rocprof_triton.sh
```

Examine the generated results:

```bash
ls rocprof_results/
cat rocprof_results/triton_analysis_summary.md
```

**Profiling Analysis:**

1. **Kernel Launch Overhead**: What is the launch overhead for Triton kernels?
2. **Memory Bandwidth**: What memory bandwidth utilization are you achieving?
3. **GPU Utilization**: How well are you utilizing the available compute units?

## Part C: Block Size Optimization (10 minutes)

### Step 6: Experiment with Block Sizes

Modify the `rmsnorm_kernel` call in `TritonRMSNorm.forward()`:

```python
# Try different block sizes
for block_size in [64, 128, 256, 512, 1024]:
    rmsnorm_kernel[grid](
        x_reshaped, self.weight, output,
        dim, self.eps, BLOCK_SIZE=block_size
    )
```

**Optimization Tasks:**

1. **Performance Testing**: Measure execution time for each block size
2. **Memory Analysis**: How does block size affect memory access patterns?
3. **Occupancy Impact**: What's the relationship between block size and GPU occupancy?

### Step 7: Memory Access Analysis

Create a simple memory access pattern analyzer:

```python
def analyze_memory_pattern():
    # Simulate memory access pattern
    dim = 2048
    block_sizes = [64, 128, 256, 512]

    for block_size in block_sizes:
        total_blocks = (dim + block_size - 1) // block_size
        print(f"Block size {block_size}: {total_blocks} blocks")

        # Analyze memory transactions
        elements_per_transaction = min(block_size, 32)  # Typical coalescing width
        transactions = (block_size + elements_per_transaction - 1) // elements_per_transaction
        print(f"  Memory transactions per block: {transactions}")
        print(f"  Total transactions: {total_blocks * transactions}")
```

**Memory Analysis Questions:**

1. **Coalescing Efficiency**: Which block size provides the best memory coalescing?
2. **Transaction Overhead**: How does the number of memory transactions scale?
3. **Cache Utilization**: What's the impact on L1/L2 cache utilization?

## Exercise Results

Document your findings:

### Performance Results Table

| Metric | Triton RMSNorm | PyTorch RMSNorm | Speedup |
|--------|----------------|------------------|---------|
| Execution Time (ms) | | | |
| Memory Usage (MB) | | | |
| Bandwidth (GB/s) | | | |

### Block Size Analysis

| Block Size | Execution Time (ms) | Memory Transactions | GPU Occupancy |
|------------|-------------------|-------------------|---------------|
| 64 | | | |
| 128 | | | |
| 256 | | | |
| 512 | | | |
| 1024 | | | |

### Key Insights

1. **Best Block Size**: _____
2. **Primary Performance Bottleneck**: _____
3. **Memory Efficiency**: _____
4. **Optimization Opportunities**: _____

## Discussion Questions

1. **Triton vs CUDA**: How does Triton kernel development compare to writing CUDA kernels?

2. **Automatic Optimizations**: What optimizations does Triton perform automatically?

3. **Performance Portability**: How portable are Triton kernels across different GPU architectures?

4. **Integration Complexity**: What are the challenges of integrating Triton kernels into PyTorch models?

## Next Steps

In Exercise 2, you'll dive deeper into the SwiGLU kernel implementation and learn about:
- Multi-dimensional memory access patterns
- Kernel fusion strategies
- Advanced optimization techniques
- Debugging Triton kernels

## Common Issues and Solutions

### Issue 1: Compilation Errors
**Problem**: Triton kernel fails to compile
**Solution**: Check that all tensor shapes are compatible and constexpr values are properly defined

### Issue 2: Performance Regression
**Problem**: Triton kernel is slower than PyTorch
**Solution**: Verify block size tuning and memory access patterns; ensure proper warmup

### Issue 3: Numerical Differences
**Problem**: Results don't match PyTorch exactly
**Solution**: Check floating-point precision and reduction order; small differences are normal

## Additional Resources

- [Triton Documentation](https://triton-lang.org/main/index.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [GPU Memory Coalescing Guide](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [ROCm Performance Guidelines](https://rocmdocs.amd.com/)