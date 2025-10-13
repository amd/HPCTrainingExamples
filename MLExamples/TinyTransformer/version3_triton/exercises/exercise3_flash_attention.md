
## Exercise 3: Flash Attention Implementation and Optimization

`exercise3_flash_attention.md` from `HPCTrainingExamples/MLExamples/TinyTransformer/version3_triton/exercises` in the Training Examples repository

**Objective**: Master advanced memory-efficient attention patterns and understand the Flash Attention algorithm implementation in Triton.

**Time**: 75 minutes

**Prerequisites**: Completed Exercises 1 and 2

### Background

Flash Attention is a memory-efficient implementation of scaled dot-product attention that:

- Reduces memory complexity from O(N^2) to O(N)
- Uses tiling to fit computations in SRAM
- Maintains numerical stability through online statistics
- Achieves significant speedups for long sequences

This exercise explores the Triton implementation and optimization strategies.

### Part A: Flash Attention Algorithm Understanding (25 minutes)

#### Step 1: Analyze the Algorithm Structure

Examine the `flash_attention_kernel` in `tiny_llama_v3.py`:

```python
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
```

**Key Components Analysis:**

1. **Tiling Strategy**: How does the algorithm tile the attention matrix?
2. **Online Statistics**: How are max values and sum exponentials maintained?
3. **Numerical Stability**: What prevents overflow in the softmax computation?

#### Step 2: Understand the Core Loop

Analyze the main computation loop:

```python
# Initialize output accumulators
output_acc = tl.zeros((BLOCK_SIZE_Q, head_dim), dtype=tl.float32)
max_scores = tl.full((BLOCK_SIZE_Q,), -float('inf'), dtype=tl.float32)
sum_exp = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)

# Process K,V blocks
for k_block_start in range(0, seq_len, BLOCK_SIZE_K):
    # Compute attention scores
    scores = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)

    # Update running statistics
    block_max = tl.max(scores, axis=1)
    new_max = tl.maximum(max_scores, block_max)
    exp_scores = tl.exp(scores - new_max[:, None])

    # Update accumulated values
    decay = tl.exp(max_scores - new_max)
    sum_exp = sum_exp * decay + tl.sum(exp_scores, axis=1)
    max_scores = new_max
```

**Algorithm Questions:**

1. **Memory Complexity**: How does this achieve O(N) memory complexity?
2. **Numerical Stability**: Why subtract the maximum before exponentiation?
3. **Online Updates**: How are the running statistics updated correctly?

#### Step 3: Compare with Standard Attention

Create a comparison analysis:

```python
def compare_attention_algorithms():
    """Compare Flash Attention with standard attention implementation."""

    print("Attention Algorithm Comparison")
    print("=" * 40)

    # Example sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    head_dim = 64

    for seq_len in seq_lengths:
        # Standard attention memory
        attention_matrix_size = seq_len * seq_len * 4  # float32
        qkv_size = 3 * seq_len * head_dim * 4
        output_size = seq_len * head_dim * 4

        standard_memory = attention_matrix_size + qkv_size + output_size

        # Flash attention memory (tiled)
        block_size = 64  # Typical block size
        tile_size = block_size * block_size * 4
        flash_memory = tile_size + qkv_size + output_size

        memory_ratio = standard_memory / flash_memory

        print(f"Seq len {seq_len:4d}: Standard {standard_memory/1e6:6.2f} MB, "
              f"Flash {flash_memory/1e6:6.2f} MB, "
              f"Ratio: {memory_ratio:5.1f}x")

    return seq_lengths, [standard_memory, flash_memory]

# Run comparison
compare_attention_algorithms()
```

#### Step 4: Analyze Causal Masking

Understand how causal masking is implemented:

```python
# Apply causal mask
causal_mask = q_offsets[:, None] >= k_offsets[None, :]
scores = tl.where(causal_mask, scores, -float('inf'))
```

**Masking Analysis:**

1. **Mask Generation**: How is the causal mask computed efficiently?
2. **Memory Impact**: What's the memory overhead of masking?
3. **Alternative Strategies**: What other masking approaches exist?

### Part B: Performance Analysis and Optimization (30 minutes)

#### Step 5: Benchmark Flash Attention Performance

Create a comprehensive benchmark:

```python
import time
import torch
import torch.nn.functional as F
from tiny_llama_v3 import TritonAttention

def benchmark_attention_implementations():
    """Benchmark Flash Attention vs standard PyTorch attention."""

    device = torch.device('cuda')

    # Test configurations
    configs = [
        (1, 8, 128, 64),    # Small
        (2, 16, 256, 64),   # Medium
        (4, 32, 512, 64),   # Large
        (2, 16, 1024, 64),  # Long sequence
        (1, 8, 2048, 64),   # Very long
    ]

    results = []

    for batch_size, num_heads, seq_len, head_dim in configs:
        print(f"\nTesting: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")

        dim = num_heads * head_dim

        # Create input
        x = torch.randn(batch_size, seq_len, dim, device=device)

        # Flash Attention (Triton)
        flash_attn = TritonAttention(dim, num_heads).to(device)

        # Standard PyTorch Attention
        class StandardAttention(torch.nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = dim // num_heads
                self.scale = 1.0 / (self.head_dim ** 0.5)

                self.q_proj = torch.nn.Linear(dim, dim, bias=False)
                self.k_proj = torch.nn.Linear(dim, dim, bias=False)
                self.v_proj = torch.nn.Linear(dim, dim, bias=False)
                self.o_proj = torch.nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                B, T, C = x.shape

                q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

                # Standard attention
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                # Causal mask
                mask = torch.tril(torch.ones(T, T, device=x.device))
                scores = scores.masked_fill(mask == 0, float('-inf'))

                attn = F.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)

                out = out.transpose(1, 2).contiguous().view(B, T, C)
                return self.o_proj(out)

        standard_attn = StandardAttention(dim, num_heads).to(device)

        # Copy weights for fair comparison
        standard_attn.q_proj.weight.data.copy_(flash_attn.q_proj.weight.data)
        standard_attn.k_proj.weight.data.copy_(flash_attn.k_proj.weight.data)
        standard_attn.v_proj.weight.data.copy_(flash_attn.v_proj.weight.data)
        standard_attn.o_proj.weight.data.copy_(flash_attn.o_proj.weight.data)

        # Benchmark Flash Attention
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(20):
            flash_output = flash_attn(x)

        torch.cuda.synchronize()
        flash_time = (time.time() - start_time) / 20

        # Benchmark Standard Attention
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(20):
            standard_output = standard_attn(x)

        torch.cuda.synchronize()
        standard_time = (time.time() - start_time) / 20

        # Memory usage
        torch.cuda.reset_peak_memory_stats()
        _ = flash_attn(x)
        flash_memory = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        _ = standard_attn(x)
        standard_memory = torch.cuda.max_memory_allocated()

        # Calculate metrics
        speedup = standard_time / flash_time
        memory_ratio = standard_memory / flash_memory
        throughput = batch_size * seq_len / flash_time

        result = {
            'config': (batch_size, num_heads, seq_len, head_dim),
            'flash_time_ms': flash_time * 1000,
            'standard_time_ms': standard_time * 1000,
            'speedup': speedup,
            'flash_memory_mb': flash_memory / 1e6,
            'standard_memory_mb': standard_memory / 1e6,
            'memory_ratio': memory_ratio,
            'throughput': throughput
        }

        results.append(result)

        print(f"  Flash Attention: {flash_time*1000:.2f} ms, {flash_memory/1e6:.1f} MB")
        print(f"  Standard Attention: {standard_time*1000:.2f} ms, {standard_memory/1e6:.1f} MB")
        print(f"  Speedup: {speedup:.2f}x, Memory reduction: {memory_ratio:.2f}x")
        print(f"  Throughput: {throughput:.0f} tokens/s")

    return results

# Run attention benchmarks
attention_results = benchmark_attention_implementations()
```

#### Step 6: Block Size Optimization

Optimize block sizes for different sequence lengths:

```python
def optimize_flash_attention_blocks():
    """Find optimal block sizes for Flash Attention."""

    device = torch.device('cuda')

    # Test different block size combinations
    block_configs = [
        (32, 32),   # Small blocks
        (64, 64),   # Medium blocks
        (128, 128), # Large blocks
        (64, 32),   # Asymmetric 1
        (32, 64),   # Asymmetric 2
        (128, 64),  # Asymmetric 3
    ]

    # Test on different sequence lengths
    seq_lengths = [256, 512, 1024]

    batch_size, num_heads, head_dim = 2, 16, 64
    dim = num_heads * head_dim

    for seq_len in seq_lengths:
        print(f"\nOptimizing for sequence length: {seq_len}")

        x = torch.randn(batch_size, seq_len, dim, device=device)

        best_time = float('inf')
        best_config = None

        for block_q, block_k in block_configs:
            # Skip if blocks are too large for sequence
            if block_q > seq_len or block_k > seq_len:
                continue

            print(f"  Testing blocks: Q={block_q}, K={block_k}")

            # Create attention with specific block sizes
            # Note: This requires modifying the kernel call
            flash_attn = TritonAttention(dim, num_heads).to(device)

            # Warmup
            for _ in range(5):
                _ = flash_attn(x)
            torch.cuda.synchronize()

            # Benchmark
            start_time = time.time()
            for _ in range(20):
                _ = flash_attn(x)
            torch.cuda.synchronize()

            avg_time = (time.time() - start_time) / 20

            print(f"    Time: {avg_time*1000:.3f} ms")

            if avg_time < best_time:
                best_time = avg_time
                best_config = (block_q, block_k)

        print(f"  Best configuration: Q={best_config[0]}, K={best_config[1]}")
        print(f"  Best time: {best_time*1000:.3f} ms")

# Run block size optimization
optimize_flash_attention_blocks()
```

#### Step 7: Memory Pattern Analysis

Analyze memory access patterns:

```python
def analyze_flash_attention_memory():
    """Analyze memory access patterns in Flash Attention."""

    print("Flash Attention Memory Pattern Analysis")
    print("=" * 45)

    # Example configuration
    batch_size, num_heads, seq_len, head_dim = 2, 16, 1024, 64
    block_q, block_k = 64, 64

    print(f"Configuration: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    print(f"Block sizes: Q={block_q}, K={block_k}")

    # Calculate memory accesses
    num_q_blocks = (seq_len + block_q - 1) // block_q
    num_k_blocks = (seq_len + block_k - 1) // block_k

    print(f"\nTiling information:")
    print(f"  Q blocks: {num_q_blocks}")
    print(f"  K blocks: {num_k_blocks}")
    print(f"  Total block pairs: {num_q_blocks * num_k_blocks}")

    # Memory per block
    q_block_size = block_q * head_dim * 4  # float32
    k_block_size = block_k * head_dim * 4
    v_block_size = block_k * head_dim * 4
    scores_size = block_q * block_k * 4

    print(f"\nMemory per block:")
    print(f"  Q block: {q_block_size/1e3:.1f} KB")
    print(f"  K block: {k_block_size/1e3:.1f} KB")
    print(f"  V block: {v_block_size/1e3:.1f} KB")
    print(f"  Scores: {scores_size/1e3:.1f} KB")
    print(f"  Total per iteration: {(q_block_size + k_block_size + v_block_size + scores_size)/1e3:.1f} KB")

    # Total memory traffic
    q_reads = num_q_blocks * q_block_size * num_k_blocks  # Q reused across K blocks
    k_reads = num_k_blocks * k_block_size * num_q_blocks  # K reused across Q blocks
    v_reads = num_k_blocks * v_block_size * num_q_blocks  # V same as K
    output_writes = seq_len * head_dim * 4

    total_traffic = q_reads + k_reads + v_reads + output_writes

    print(f"\nTotal memory traffic:")
    print(f"  Q reads: {q_reads/1e6:.2f} MB")
    print(f"  K reads: {k_reads/1e6:.2f} MB")
    print(f"  V reads: {v_reads/1e6:.2f} MB")
    print(f"  Output writes: {output_writes/1e6:.2f} MB")
    print(f"  Total: {total_traffic/1e6:.2f} MB")

    # Compare with standard attention
    standard_traffic = (
        3 * seq_len * head_dim * 4 +  # Q, K, V
        seq_len * seq_len * 4 +       # Attention matrix
        seq_len * head_dim * 4        # Output
    )

    print(f"\nStandard attention traffic: {standard_traffic/1e6:.2f} MB")
    print(f"Flash attention reduction: {standard_traffic/total_traffic:.2f}x")

    return {
        'flash_traffic_mb': total_traffic / 1e6,
        'standard_traffic_mb': standard_traffic / 1e6,
        'reduction_ratio': standard_traffic / total_traffic
    }

# Run memory analysis
memory_analysis = analyze_flash_attention_memory()
```

### Part C: Advanced Optimizations and Debugging (20 minutes)

#### Step 8: Numerical Stability Testing

Test numerical stability across different conditions:

```python
def test_numerical_stability():
    """Test numerical stability of Flash Attention implementation."""

    device = torch.device('cuda')

    # Test conditions
    test_cases = [
        ("normal", 1.0, 0.0),
        ("large_values", 10.0, 0.0),
        ("small_values", 0.1, 0.0),
        ("extreme_large", 100.0, 0.0),
        ("with_noise", 1.0, 0.1),
    ]

    batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64
    dim = num_heads * head_dim

    flash_attn = TritonAttention(dim, num_heads).to(device)

    for name, scale, noise in test_cases:
        print(f"\nTesting {name} (scale={scale}, noise={noise}):")

        # Generate test input
        x = torch.randn(batch_size, seq_len, dim, device=device) * scale
        if noise > 0:
            x += torch.randn_like(x) * noise

        try:
            output = flash_attn(x)

            # Check for NaN/Inf
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()

            print(f"  Input range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")

            if has_nan or has_inf:
                print("  WARNING: Numerical instability detected!")
            else:
                print("  PASS Numerically stable")

        except Exception as e:
            print(f"  FAIL Error: {e}")

# Run stability tests
test_numerical_stability()
```

#### Step 9: Performance Profiling Integration

Integrate with ROCProfiler for detailed analysis:

```python
def create_flash_attention_profile():
    """Create focused profiling for Flash Attention kernels."""

    # Create ROCProfiler configuration for Flash Attention
    profile_config = """
# Flash Attention Kernel Profiling Configuration
pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts
pmc : VALUUtilization FlatVMemUtilization MemUnitBusy L2CacheHit
pmc : WriteUnitStalled ALUStalledByLDS LDSBankConflict
range: 0x1000000000000:0x2000000000000
gpu: 0
kernel: flash_attention_kernel
"""

    with open("flash_attention_profile.txt", "w") as f:
        f.write(profile_config)

    print("Created Flash Attention profiling configuration")
    print("Run with: rocprof --input flash_attention_profile.txt python3 tiny_llama_v3.py")

# Create profiling configuration
create_flash_attention_profile()
```

### Exercise Results

#### Performance Summary Table

| Sequence Length | Flash Attention (ms) | Standard Attention (ms) | Speedup | Memory Reduction |
|----------------|---------------------|------------------------|---------|------------------|
| 128 | | | | |
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |
| 2048 | | | | |

#### Block Size Optimization Results

| Sequence Length | Optimal Q Block | Optimal K Block | Best Time (ms) | Notes |
|----------------|----------------|----------------|----------------|-------|
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |

#### Memory Analysis Results

- **Flash Attention Memory**: _____ MB
- **Standard Attention Memory**: _____ MB
- **Memory Reduction**: _____x
- **Arithmetic Intensity**: _____ FLOPs/byte

#### Key Insights

1. **Performance Scaling**: How does Flash Attention performance scale with sequence length?
2. **Memory Efficiency**: What's the memory reduction at different sequence lengths?
3. **Optimal Block Sizes**: What patterns emerge in optimal block size selection?
4. **Numerical Stability**: Are there any stability concerns with the implementation?

### Discussion Questions

1. **Algorithm Trade-offs**: What are the trade-offs between memory efficiency and computational complexity in Flash Attention?

2. **Implementation Challenges**: What are the main challenges in implementing Flash Attention in Triton vs CUDA?

3. **Sequence Length Scaling**: How does the algorithm's efficiency change with very long sequences (8K, 16K tokens)?

4. **Hardware Considerations**: How might different GPU architectures affect Flash Attention performance?

### Next Steps

With Version 3 complete, you've learned:
- Advanced Triton kernel development
- Memory-efficient algorithm implementation
- Performance optimization strategies
- Numerical stability considerations

Version 4 will cover ultra-fused implementations combining all optimizations into a single, highly optimized kernel suite.

### Troubleshooting Guide

#### Common Issues

1. **Kernel Compilation Errors**
   - Check tensor dimension compatibility
   - Verify block sizes don't exceed hardware limits
   - Ensure proper constexpr usage

2. **Performance Regression**
   - Verify block sizes are optimal for your sequence length
   - Check memory access patterns
   - Ensure proper warmup before benchmarking

3. **Numerical Instability**
   - Monitor for overflow in softmax computation
   - Check running statistics update logic
   - Verify causal mask application

4. **Memory Issues**
   - Reduce block sizes if running out of memory
   - Check for memory leaks in repeated runs
   - Monitor peak memory usage during profiling

