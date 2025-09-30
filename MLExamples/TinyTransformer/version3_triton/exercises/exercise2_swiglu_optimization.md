# Exercise 2: SwiGLU Kernel Optimization

**Objective**: Master advanced Triton kernel development through SwiGLU optimization and learn multi-dimensional memory access patterns.

**Time**: 60 minutes

**Prerequisites**: Completed Exercise 1

## Background

The SwiGLU (Swish-Gated Linear Unit) is a key component in modern transformer architectures. It combines:
- Gate projection with SiLU activation
- Up projection
- Element-wise multiplication
- Down projection

Traditional implementations require multiple kernel launches and intermediate storage. Our Triton kernel fuses the gate and up projections with activation, reducing memory traffic and improving performance.

## Part A: SwiGLU Kernel Deep Dive (20 minutes)

### Step 1: Analyze the Kernel Structure

Examine the `swiglu_kernel` in `tiny_llama_v3.py`:

```python
@triton.jit
def swiglu_kernel(
    x_ptr, gate_weight_ptr, up_weight_ptr, output_ptr,
    batch_size, seq_len, d_model, d_ff,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
```

**Analysis Questions:**

1. **Multi-dimensional Blocking**: Why does this kernel use three different block sizes?
2. **Memory Layout**: How are the tensors laid out in memory (batch, sequence, feature dimensions)?
3. **Compute Intensity**: What is the arithmetic intensity of this kernel?

### Step 2: Understand the Computation Flow

Follow the kernel execution:

```python
# Load input
input_offset = batch_idx * seq_len * d_model + seq_idx * d_model
x_block = tl.load(x_ptr + input_offset + tl.arange(0, d_model))

# Compute projections
for i in range(0, d_model, BLOCK_SIZE_D):
    x_vals = tl.load(x_ptr + input_offset + i + tl.arange(0, BLOCK_SIZE_D))
    gate_weights = tl.load(gate_weight_ptr + d_idx * d_model + i + tl.arange(0, BLOCK_SIZE_D))
    up_weights = tl.load(up_weight_ptr + d_idx * d_model + i + tl.arange(0, BLOCK_SIZE_D))

    gate_sum += tl.sum(x_vals * gate_weights)
    up_sum += tl.sum(x_vals * up_weights)

# Apply activation
gate_activated = gate_sum / (1.0 + tl.exp(-gate_sum))
result = gate_activated * up_sum
```

**Computation Analysis:**

1. **Memory Reuse**: How does the kernel maximize input data reuse?
2. **Reduction Pattern**: Explain the dot product computation strategy
3. **Activation Fusion**: How is the SiLU activation integrated efficiently?

### Step 3: Memory Access Pattern Visualization

Create a visualization tool for memory access patterns:

```python
def visualize_swiglu_access_pattern():
    """Visualize memory access patterns for SwiGLU kernel."""

    # Example dimensions
    batch_size, seq_len, d_model, d_ff = 2, 4, 8, 12

    print("SwiGLU Memory Access Pattern Analysis")
    print("=" * 50)

    print(f"Tensor shapes:")
    print(f"  Input (x): [{batch_size}, {seq_len}, {d_model}]")
    print(f"  Gate weights: [{d_ff}, {d_model}]")
    print(f"  Up weights: [{d_ff}, {d_model}]")
    print(f"  Output: [{batch_size}, {seq_len}, {d_ff}]")

    print(f"\nTotal elements:")
    print(f"  Input: {batch_size * seq_len * d_model}")
    print(f"  Weights: {2 * d_ff * d_model}")
    print(f"  Output: {batch_size * seq_len * d_ff}")

    # Analyze memory traffic
    input_reads = batch_size * seq_len * d_model * d_ff  # Each input element read d_ff times
    weight_reads = 2 * d_ff * d_model * batch_size * seq_len  # Weight reuse across batch/seq
    output_writes = batch_size * seq_len * d_ff

    total_bytes = (input_reads + weight_reads + output_writes) * 4  # float32

    print(f"\nMemory traffic analysis:")
    print(f"  Input reads: {input_reads}")
    print(f"  Weight reads: {weight_reads}")
    print(f"  Output writes: {output_writes}")
    print(f"  Total memory traffic: {total_bytes / 1e6:.2f} MB")

    # Compute to memory ratio
    flops = 2 * batch_size * seq_len * d_model * d_ff * 2  # 2 projections, 2 ops per MAC
    arithmetic_intensity = flops / total_bytes * 4  # ops per byte

    print(f"  FLOPs: {flops}")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.2f} ops/byte")

# Run the analysis
visualize_swiglu_access_pattern()
```

## Part B: Performance Optimization (25 minutes)

### Step 4: Block Size Tuning

Create a systematic block size tuning script:

```python
import time
import torch
from tiny_llama_v3 import TritonSwiGLU

def tune_swiglu_block_sizes():
    """Tune block sizes for optimal SwiGLU performance."""

    device = torch.device('cuda')
    batch_size, seq_len, d_model = 4, 512, 2048
    hidden_dim = int(2.67 * d_model)

    # Test different block size combinations
    block_configs = [
        (1, 1, 32),   # Small blocks
        (1, 1, 64),   # Medium blocks
        (1, 1, 128),  # Large blocks
        (1, 2, 64),   # Sequence blocking
        (2, 1, 64),   # Batch blocking
        (1, 1, 256),  # Extra large feature blocks
    ]

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    results = []

    for b_block, s_block, d_block in block_configs:
        print(f"\nTesting block configuration: B={b_block}, S={s_block}, D={d_block}")

        # Create modified SwiGLU with specific block sizes
        swiglu = TritonSwiGLU(d_model, hidden_dim).to(device)

        # Warmup
        for _ in range(10):
            _ = swiglu(x)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(100):
            output = swiglu(x)
        torch.cuda.synchronize()

        avg_time = (time.time() - start_time) / 100

        results.append({
            'config': (b_block, s_block, d_block),
            'time_ms': avg_time * 1000,
            'throughput': batch_size * seq_len / avg_time
        })

        print(f"  Time: {avg_time*1000:.3f} ms")
        print(f"  Throughput: {batch_size * seq_len / avg_time:.0f} tokens/s")

    # Find best configuration
    best_result = min(results, key=lambda x: x['time_ms'])
    print(f"\nBest configuration: {best_result['config']}")
    print(f"Best time: {best_result['time_ms']:.3f} ms")

    return results

# Run block size tuning
block_results = tune_swiglu_block_sizes()
```

### Step 5: Memory Layout Optimization

Experiment with different memory layouts:

```python
def analyze_memory_layouts():
    """Analyze impact of different memory layouts on performance."""

    device = torch.device('cuda')
    batch_size, seq_len, d_model = 4, 512, 2048
    hidden_dim = int(2.67 * d_model)

    # Test different weight layouts
    layouts = ['row_major', 'column_major', 'transposed']

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    for layout in layouts:
        print(f"\nTesting {layout} weight layout:")

        swiglu = TritonSwiGLU(d_model, hidden_dim).to(device)

        if layout == 'column_major':
            # Transpose weights for column-major access
            swiglu.gate_proj.weight.data = swiglu.gate_proj.weight.data.t().contiguous().t()
            swiglu.up_proj.weight.data = swiglu.up_proj.weight.data.t().contiguous().t()
        elif layout == 'transposed':
            # Use transposed weights
            swiglu.gate_proj.weight.data = swiglu.gate_proj.weight.data.t().contiguous()
            swiglu.up_proj.weight.data = swiglu.up_proj.weight.data.t().contiguous()

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(50):
            output = swiglu(x)

        torch.cuda.synchronize()
        avg_time = (time.time() - start_time) / 50

        print(f"  Average time: {avg_time*1000:.3f} ms")
        print(f"  Memory bandwidth: {estimate_bandwidth(x, swiglu, avg_time):.1f} GB/s")

def estimate_bandwidth(x, swiglu, exec_time):
    """Estimate memory bandwidth utilization."""

    # Calculate memory footprint
    input_size = x.numel() * 4  # float32
    weight_size = (swiglu.gate_proj.weight.numel() + swiglu.up_proj.weight.numel()) * 4
    output_size = x.shape[0] * x.shape[1] * swiglu.gate_proj.out_features * 4

    total_bytes = input_size + weight_size + output_size
    bandwidth = total_bytes / exec_time / 1e9

    return bandwidth

# Run memory layout analysis
analyze_memory_layouts()
```

### Step 6: Arithmetic Intensity Analysis

Calculate and optimize arithmetic intensity:

```python
def analyze_arithmetic_intensity():
    """Analyze arithmetic intensity and roofline performance."""

    batch_size, seq_len, d_model = 4, 512, 2048
    hidden_dim = int(2.67 * d_model)

    # Calculate FLOPs
    # Gate projection: batch_size * seq_len * d_model * hidden_dim * 2 (MAC)
    gate_flops = batch_size * seq_len * d_model * hidden_dim * 2

    # Up projection: same as gate
    up_flops = gate_flops

    # SiLU activation: ~4 FLOPs per element (exp, add, div, mul)
    silu_flops = batch_size * seq_len * hidden_dim * 4

    # Element-wise multiply: 1 FLOP per element
    multiply_flops = batch_size * seq_len * hidden_dim

    total_flops = gate_flops + up_flops + silu_flops + multiply_flops

    # Calculate memory traffic
    input_bytes = batch_size * seq_len * d_model * 4
    gate_weight_bytes = d_model * hidden_dim * 4
    up_weight_bytes = d_model * hidden_dim * 4
    output_bytes = batch_size * seq_len * hidden_dim * 4

    total_bytes = input_bytes + gate_weight_bytes + up_weight_bytes + output_bytes

    arithmetic_intensity = total_flops / total_bytes

    print("SwiGLU Arithmetic Intensity Analysis")
    print("=" * 40)
    print(f"Problem size: {batch_size}x{seq_len}x{d_model} -> {hidden_dim}")
    print(f"Total FLOPs: {total_flops/1e9:.2f} GFLOPs")
    print(f"Total memory: {total_bytes/1e6:.2f} MB")
    print(f"Arithmetic intensity: {arithmetic_intensity:.2f} FLOPs/byte")

    # Roofline analysis
    peak_flops = 200e12  # Example: 200 TFLOPS (MI250X)
    peak_bandwidth = 1600e9  # Example: 1.6 TB/s

    compute_bound_intensity = peak_flops / peak_bandwidth

    print(f"\nRoofline Analysis:")
    print(f"Peak compute: {peak_flops/1e12:.0f} TFLOPS")
    print(f"Peak bandwidth: {peak_bandwidth/1e9:.0f} GB/s")
    print(f"Compute-bound threshold: {compute_bound_intensity:.2f} FLOPs/byte")

    if arithmetic_intensity > compute_bound_intensity:
        print("Kernel is compute-bound - optimize arithmetic operations")
        bottleneck = "compute"
    else:
        print("Kernel is memory-bound - optimize memory access")
        bottleneck = "memory"

    return {
        'arithmetic_intensity': arithmetic_intensity,
        'total_flops': total_flops,
        'total_bytes': total_bytes,
        'bottleneck': bottleneck
    }

# Run arithmetic intensity analysis
intensity_results = analyze_arithmetic_intensity()
```

## Part C: Advanced Optimization Techniques (15 minutes)

### Step 7: Implement Kernel Variants

Create optimized kernel variants:

```python
# Version 1: Basic implementation (current)
# Version 2: Optimized for memory-bound workloads
# Version 3: Optimized for compute-bound workloads

@triton.jit
def swiglu_kernel_optimized_memory(
    x_ptr, gate_weight_ptr, up_weight_ptr, output_ptr,
    batch_size, seq_len, d_model, d_ff,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Memory-optimized SwiGLU kernel with better data reuse."""

    # Single thread processes entire token
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    input_offset = batch_idx * seq_len * d_model + seq_idx * d_model

    # Process all outputs for this token
    for d_out in range(0, d_ff, BLOCK_SIZE_D):
        gate_sum = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
        up_sum = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

        # Load output indices
        d_indices = d_out + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_indices < d_ff

        # Compute projections
        for d_in in range(d_model):
            x_val = tl.load(x_ptr + input_offset + d_in)

            gate_weights = tl.load(gate_weight_ptr + d_indices * d_model + d_in, mask=d_mask)
            up_weights = tl.load(up_weight_ptr + d_indices * d_model + d_in, mask=d_mask)

            gate_sum += x_val * gate_weights
            up_sum += x_val * up_weights

        # Apply SiLU and multiply
        gate_activated = gate_sum / (1.0 + tl.exp(-gate_sum))
        result = gate_activated * up_sum

        # Store results
        output_offset = batch_idx * seq_len * d_ff + seq_idx * d_ff + d_indices
        tl.store(output_ptr + output_offset, result, mask=d_mask)


def benchmark_kernel_variants():
    """Benchmark different kernel implementations."""

    device = torch.device('cuda')
    batch_size, seq_len, d_model = 4, 512, 2048
    hidden_dim = int(2.67 * d_model)

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    variants = [
        ('Original', TritonSwiGLU(d_model, hidden_dim)),
        # Add other variants here
    ]

    for name, swiglu in variants:
        swiglu = swiglu.to(device)

        # Warmup
        for _ in range(10):
            _ = swiglu(x)
        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(100):
            output = swiglu(x)
        torch.cuda.synchronize()

        avg_time = (time.time() - start_time) / 100
        print(f"{name}: {avg_time*1000:.3f} ms")

# Run variant benchmarks
benchmark_kernel_variants()
```

## Exercise Results

### Performance Comparison Table

| Configuration | Time (ms) | Speedup vs PyTorch | Memory Usage | Bandwidth (GB/s) |
|---------------|-----------|-------------------|--------------|------------------|
| Original SwiGLU | | | | |
| Block Size (1,1,32) | | | | |
| Block Size (1,1,64) | | | | |
| Block Size (1,1,128) | | | | |
| Memory Optimized | | | | |

### Arithmetic Intensity Analysis

- **Total FLOPs**: _____ GFLOPs
- **Memory Traffic**: _____ MB
- **Arithmetic Intensity**: _____ FLOPs/byte
- **Performance Bottleneck**: _____ (compute/memory)
- **Optimization Strategy**: _____

### Key Findings

1. **Optimal Block Size**: _____
2. **Memory Layout Impact**: _____
3. **Arithmetic Intensity**: _____
4. **Performance Bottleneck**: _____

## Discussion Questions

1. **Multi-dimensional Blocking**: How do you choose optimal block sizes for multi-dimensional problems?

2. **Memory vs Compute Optimization**: When should you optimize for memory bandwidth vs computational throughput?

3. **Kernel Fusion Trade-offs**: What are the trade-offs between kernel fusion and memory usage?

4. **Scalability**: How do these optimizations scale with different problem sizes?

## Next Steps

Exercise 3 will cover Flash Attention implementation, focusing on:
- Memory-efficient attention patterns
- Tiling strategies for large sequences
- Numerical stability in custom kernels
- Advanced debugging techniques