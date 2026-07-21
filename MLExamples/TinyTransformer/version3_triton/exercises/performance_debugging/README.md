# Performance Debugging Exercise: V3 Triton Optimization Journey

## Overview

This exercise demonstrates the systematic debugging and optimization process for V3 Triton kernels. You'll learn how to:

1. **Diagnose incorrect model behavior** (wrong loss values)
2. **Fix correctness issues** (weight initialization)
3. **Profile and identify performance bottlenecks**
4. **Systematically optimize for performance**

## The Problem

Initial V3 implementation showed:
-  **Loss = 942** (should be ~7 like V1/V2)
-  **Fake timing** (reported 4ms but actually much slower)
-  **6.4x slower than baseline** after initial fixes

## Exercise Progression

Each file represents a stage in the debugging process:

### Stage 1: Broken Loss (`v3_stage1_broken_loss.py`)
**Problem:** Loss = 942 instead of ~7
**Root Cause:** Missing weight initialization
**What to Learn:**
- How to add diagnostic logging
- How to trace values through the model
- How exploding logits break training

**Run:**
```bash
python v3_stage1_broken_loss.py --batch-size 8 --seq-len 128 --num-steps 20
```

**Expected Output:**
```
Loss: 942.8047  # WRONG!
Logits stats: min=-161, max=1025, std=43.79  # Exploding values
```

---

### Stage 2: Fixed Loss, Terrible Performance (`v3_stage2_slow_performance.py`)
**Problem:** Loss fixed (7.0) but only 15.2 samples/sec (vs V1's 97 samples/sec)
**Root Cause:** Non-contiguous tensors after `repeat_interleave` for GQA
**What to Learn:**
- How memory layout affects Triton kernel performance
- Why `.contiguous()` matters for GPU kernels
- How to identify stride-related issues

**Run:**
```bash
python v3_stage2_slow_performance.py --batch-size 8 --seq-len 128 --num-steps 20
```

**Expected Output:**
```
Loss: 7.0108  # CORRECT!
Speed: 15.2 samples/sec  # TERRIBLE! (V1 = 97 samples/sec)
Time: 526ms per batch
```

---

### Stage 3: Better Performance, Wrong Timing (`v3_stage3_fake_timing.py`)
**Problem:** Improved to 310 samples/sec but timing breakdown is wrong
**Root Cause:** Missing CUDA synchronization for individual operation timing
**What to Learn:**
- GPU operations are asynchronous
- How to properly measure GPU kernel timing
- Why you need `torch.cuda.synchronize()`

**Run:**
```bash
python v3_stage3_fake_timing.py --batch-size 8 --seq-len 128 --num-steps 20
```

**Expected Output:**
```
Loss: 7.0108  # CORRECT!
Speed: 310.8 samples/sec  # GOOD!
Forward: 3.2ms  # Seems reasonable
Backward: 0.2ms  # WRONG! Too fast!
Total: 25.7ms  # Doesn't add up! (3.2 + 0.2 + 0.2 ≠ 25.7)
```

---

### Stage 4: Accurate Timing, Slow Kernels (`v3_stage4_slow_kernels.py`)
**Problem:** Accurate timing shows forward pass is 25.5ms (2.4x slower than V1's 10.8ms)
**Root Cause:** Inefficient Triton SwiGLU kernel doing manual matrix multiplication
**What to Learn:**
- How to identify kernel bottlenecks
- When NOT to use custom kernels (for large matrix ops)
- Why PyTorch BLAS is faster than naive Triton implementations

**Run:**
```bash
python v3_stage4_slow_kernels.py --batch-size 8 --seq-len 128 --num-steps 20
```

**Expected Output:**
```
Loss: 7.0108  # CORRECT!
Speed: 305.9 samples/sec  # STILL SLOWER THAN V1!
Forward: 25.5ms  # TOO SLOW! (V1 = 10.8ms)
Backward: 0.3ms
Total: 26.2ms
```

**Profiling Analysis:**
- SwiGLU kernel launches 2,097,152 threads (batch × seq × d_ff = 8 × 128 × 2048)
- Each thread does manual reduction over 512 dimensions
- PyTorch's optimized BLAS would be much faster

---

### Stage 5: Final Optimized (`../tiny_llama_v3.py`)
**Solution:** Use PyTorch for matrix multiplies, Triton only for element-wise fusion
**Result:** 2065 samples/sec (5.5x faster than V1!)
**What to Learn:**
- Hybrid optimization: use the best tool for each operation
- When to use Triton (memory-bound ops, fusion opportunities)
- When to use PyTorch (compute-bound large matrix ops)

**Run:**
```bash
cd .. && python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20
```

**Expected Output:**
```
Loss: 7.0108  # CORRECT!
Speed: 2065.0 samples/sec  # EXCELLENT! (5.5x faster than V1!)
Forward: 3.2ms  # Fast!
Backward: 0.3ms
Total: 3.9ms  # Dramatic improvement!
Memory: 281.8 MB  # 46% less than V1's 522.3 MB
```

---

## Profiling with ROCm Tools

### Using rocprof to Profile Each Stage

For each stage, you can generate detailed profiling traces:

```bash
# Stage 1: Broken Loss (short run to see the issue)
rocprof --stats -o stage1_broken.csv python v3_stage1_broken_loss.py --batch-size 8 --seq-len 128 --num-steps 5

# Stage 2: Slow Performance
rocprof --stats -o stage2_slow.csv python v3_stage2_slow_performance.py --batch-size 8 --seq-len 128 --num-steps 20

# Stage 4: Slow Kernels (shows SwiGLU bottleneck)
rocprof --stats -o stage4_kernels.csv python v3_stage4_slow_kernels.py --batch-size 8 --seq-len 128 --num-steps 20

# Stage 5: Final Optimized
rocprof --stats -o stage5_optimized.csv python ../tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20
```

### What to Look for in Traces

**Stage 2 (Slow Performance):**
- Look for non-coalesced memory accesses in Flash Attention kernel
- High L2 cache miss rate
- Memory stalls

**Stage 4 (Slow Kernels):**
- SwiGLU kernel shows:
  - 2M+ kernel launches
  - Low occupancy (< 25%)
  - High kernel launch overhead
- Compare to PyTorch matmul:
  - Uses rocBLAS (optimized)
  - High throughput (90%+ of peak)

**Stage 5 (Optimized):**
- Flash Attention: High occupancy, good memory throughput
- RMSNorm: Fused operations, low latency
- Matrix ops: Delegated to rocBLAS (optimal)

### Analyzing with rocprofv2

For more detailed analysis:

```bash
# Profile with kernel trace
rocprofv2 --kernel-trace -o stage4_trace.json python v3_stage4_slow_kernels.py --batch-size 8 --seq-len 128 --num-steps 10

# View in Perfetto UI
# Upload stage4_trace.json to https://ui.perfetto.dev
```

**What to observe:**
- Kernel timeline showing SwiGLU dominating execution
- Memory transfer patterns
- Kernel duration vs. compute capability

---

## Key Learnings

### 1. Correctness First, Performance Second
- Stage 1 shows why: broken model can't be optimized
- Always validate loss/accuracy before optimizing

### 2. Systematic Debugging
- Add diagnostic logging (Stage 1)
- Measure accurately (Stage 3)
- Profile to identify bottlenecks (Stage 4)
- Fix one issue at a time

### 3. Know Your Tools
- **Triton**: Memory-bound ops, element-wise fusion, Flash Attention
- **PyTorch/BLAS**: Compute-bound matrix operations
- **Profilers**: rocprof for GPU metrics, timing for coarse analysis

### 4. Common Performance Pitfalls
-  **Tensor contiguity**: Always `.contiguous()` before Triton kernels
-  **CUDA synchronization**: Required for accurate GPU timing
-  **Kernel granularity**: Avoid launching millions of tiny kernels
-  **Use optimized libraries**: Don't reimplement BLAS in Triton

### 5. Optimization is Iterative
- V1 baseline: 372.9 samples/sec
- Stage 2 (correct): 15.2 samples/sec (40x SLOWER!)
- Stage 3 (contiguous): 310.8 samples/sec (0.83x baseline)
- **Stage 5 (optimized): 2065.0 samples/sec (5.5x FASTER!)**

---

## Exercises

### Exercise 1: Diagnose Stage 1
Run `v3_stage1_broken_loss.py` and:
1. Uncomment the diagnostic logging
2. Identify which layer produces exploding values
3. Explain why default weight initialization causes this

### Exercise 2: Profile Stage 2
1. Run with rocprof: `rocprof --stats python v3_stage2_slow_performance.py ...`
2. Find the Flash Attention kernel in the trace
3. Look at memory metrics - what's wrong?

### Exercise 3: Compare Stage 4 vs Stage 5
1. Profile both versions with rocprof
2. Compare SwiGLU execution time
3. Explain the 8x speedup in the forward pass

### Exercise 4: Design Your Own Optimization
1. Look at the RMSNorm kernel implementation
2. Can you further optimize it?
3. What profiling metrics would validate your optimization?

---

## Next Steps

After completing this exercise:

1. **Apply to V4**: V4 has similar issues - can you fix them?
2. **Custom Kernels**: Try writing your own Triton kernel for a simple operation
3. **Advanced Profiling**: Learn rocprofv2 for detailed analysis
4. **Production Deployment**: Consider hybrid Triton+PyTorch approaches

---

## Additional Resources

- **Triton Documentation**: https://triton-lang.org/
- **ROCm Profiling Guide**: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
- **Flash Attention Paper**: https://arxiv.org/abs/2205.14135
- **PyTorch Profiler**: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

---

## Summary Table

| Stage | Loss | Speed (samples/sec) | Issue | Fix |
|-------|------|---------------------|-------|-----|
| 1 |  942 | N/A | Missing weight init | Add `_init_weights()` |
| 2 |  7.0 |  15.2 | Non-contiguous tensors | Add `.contiguous()` |
| 3 |  7.0 |  310.8 | Wrong timing | Add CUDA sync |
| 4 |  7.0 |  305.9 | Slow Triton SwiGLU | Use PyTorch matmul |
| 5 |  7.0 |  2065.0 | **OPTIMIZED!** | Hybrid approach |

**Baseline (V1):** 372.9 samples/sec
**Final Speedup:** 5.5x faster, 46% less memory
