
# Version 3: Triton Kernel Integration - Workshop Edition

`README_WORKSHOP.md` from `HPCTrainingExamples/MLExamples/TinyTransformer/version3_triton` in the Training Examples repository

**Objective**: Implement custom GPU kernels using Triton for maximum performance optimization

**Actual Performance**: **5.5x speedup** over baseline, **46% memory reduction**

**Learning Focus**: GPU kernel programming, performance debugging, hybrid optimization strategies

---

##  Quick Start (5 minutes)

```bash
cd version3_triton/

# Run the optimized version
python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20

# Expected output:
# Loss: 7.0108  (correct!)
# Speed: 2065.0 samples/sec  (5.5x faster than V1!)
# Memory: 281.8 MB  (46% less than V1's 522 MB!)
```

---

##  Performance Results

### Actual Measurements (AMD MI325X, ROCm 6.4.4)

**Test Configuration:** Batch=8, SeqLen=128, Hidden=512, Layers=8, Heads=8

| Metric | V1 Baseline | V3 Optimized | Improvement |
|--------|-------------|--------------|-------------|
| **Training Speed** | 372.9 samples/sec | **2065.0 samples/sec** | **5.5x faster** |
| **Batch Time** | 21.7 ms | **3.9 ms** | **5.6x faster** |
| **Forward Pass** | 10.8 ms | **3.2 ms** | **3.4x faster** |
| **Backward Pass** | 9.2 ms | **0.3 ms** | **30x faster** |
| **Memory Usage** | 522.3 MB | **281.8 MB** | **46% reduction** |
| **Throughput** | 47,735 tokens/sec | **264,320 tokens/sec** | **5.5x faster** |

---

##  Key Concepts

### What is Triton?

**Triton** is a Python-based GPU programming language that makes it easy to write high-performance GPU kernels without dealing with low-level CUDA/HIP complexity.

**Why Use Triton?**
-  Python-like syntax (easier than CUDA/HIP)
-  Automatic memory coalescing and optimization
-  Works on both NVIDIA and AMD GPUs
-  Great for memory-bound operations and fusion

**When NOT to Use Triton?**
-  Large matrix multiplications (use PyTorch/rocBLAS instead)
-  Operations already well-optimized in PyTorch
-  Compute-bound ops where BLAS libraries excel

---

##  Optimizations Applied in V3

### 1. Flash Attention (Triton Kernel)
**What it does:** Memory-efficient attention using online softmax

**PyTorch Standard Attention:**
```python
# Materializes full attention matrix: O(N²) memory
scores = Q @ K.T  # [batch, heads, seq, seq] - HUGE!
attn = softmax(scores)
output = attn @ V
```

**Flash Attention:**
```python
# Online computation: O(N) memory
# Processes attention in blocks, never materializes full matrix
# Uses tiled computation with recomputation for backward pass
```

**Result:**
- 46% memory reduction (282 MB vs 522 MB)
- Enables longer sequences
- Slightly faster forward pass

### 2. RMSNorm (Triton Kernel)
**What it does:** Fused variance computation + normalization

**Before (PyTorch):** 3 separate kernels
```python
variance = x.pow(2).mean(-1, keepdim=True)  # Kernel 1
rstd = torch.rsqrt(variance + eps)           # Kernel 2
output = (x * rstd) * weight                 # Kernel 3
```

**After (Triton):** Single fused kernel
```python
# All operations in one kernel launch
# Variance computed in registers
# Immediate normalization and scaling
```

**Result:**
- 3x fewer kernel launches
- Better cache utilization
- Reduced memory bandwidth

### 3. Hybrid SwiGLU Strategy
**Critical Lesson:** Don't use custom kernels for everything!

**Initial (Broken) Approach:**
```python
# Used Triton kernel for matrix multiply - BAD IDEA!
# Launched 2,097,152 threads (batch × seq × d_ff)
# Each thread did manual reduction - VERY SLOW!
# Result: 25.5ms forward pass (2.4x SLOWER than V1!)
```

**Optimized (Hybrid) Approach:**
```python
# Use PyTorch for matrix multiplies (rocBLAS optimized)
gate = self.gate_proj(x)  # rocBLAS
up = self.up_proj(x)       # rocBLAS

# Use PyTorch for activation (already fused)
gate_activated = F.silu(gate) * up

# Use PyTorch for final projection
output = self.down_proj(intermediate)  # rocBLAS
```

**Result:**
- 8x forward pass speedup (25.5ms → 3.2ms)
- **Key insight:** Use the best tool for each operation

### 4. Tensor Contiguity (Critical!)
**The Bug:** Non-contiguous tensors after `repeat_interleave` for GQA

**Before:**
```python
k = k.repeat_interleave(n_rep, dim=1)  # Creates non-contiguous tensor!
v = v.repeat_interleave(n_rep, dim=1)  # Bad memory layout for Triton!
```

**After:**
```python
k = k.repeat_interleave(n_rep, dim=1).contiguous()  # Fix memory layout
v = v.repeat_interleave(n_rep, dim=1).contiguous()  # Now Triton-friendly!
```

**Result:**
- 20x speedup! (15.2 → 310.8 samples/sec)
- Triton kernels depend on contiguous memory for efficient access
- Always check tensor contiguity before passing to custom kernels

### 5. Proper Weight Initialization
**The Bug:** Default `nn.Embedding` uses `Normal(0, 1)` - too large!

**Before:**
```python
# No weight initialization
# Embedding weight ~ Normal(0, 1)
# With dim=1024, logits have std ≈ √1024 ≈ 32
# Result: Logits explode to hundreds, loss = 942!
```

**After:**
```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**Result:**
- Loss: 942 → 7.0
- Critical for tied weights (embedding + lm_head)
- Small std prevents exploding gradients

---

##  Performance Debugging Exercise

Want to see the complete optimization journey? Try our hands-on debugging exercise:

```bash
cd exercises/performance_debugging/

# Read the guide
cat README.md

# Run all 5 stages of optimization with profiling
./run_all_stages.sh

# This shows the complete journey:
# Stage 1: Broken (loss=942) - missing weight init
# Stage 2: Slow (15 samp/s) - non-contiguous tensors
# Stage 3: Better (311 samp/s) - added .contiguous()
# Stage 4: Same (306 samp/s) - accurate timing revealed issue
# Stage 5: Optimal (2065 samp/s) - hybrid kernel strategy!
```

**What you'll learn:**
- How to diagnose incorrect model behavior (exploding loss)
- How to identify performance bottlenecks with profiling
- When to use custom kernels vs. optimized libraries
- How memory layout affects GPU performance
- Systematic debugging methodology

---

##  Profiling Commands

### Basic Profiling
```bash
# Run with basic stats
rocprof --stats python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 10

# Output: CSV file with kernel execution times
```

### Detailed Kernel Trace
```bash
# Generate detailed timeline trace
rocprofv2 --kernel-trace -o v3_trace.json python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 5

# View at https://ui.perfetto.dev
```

### What to Look For in Profiles
1. **Kernel Execution Time:** Which kernels take the most time?
2. **Memory Bandwidth:** Are you memory-bound or compute-bound?
3. **Occupancy:** How many wavefronts/warps are active?
4. **Launch Overhead:** Too many small kernel launches?

---

##  Key Learnings

### 1. Correctness First, Performance Second
- Stage 1 had broken loss (942 instead of 7)
- No point optimizing a broken model!
- Always validate correctness before optimizing

### 2. Memory Layout Matters
- Non-contiguous tensors killed performance (20x slower!)
- Always `.contiguous()` before Triton kernels
- Check with `tensor.is_contiguous()`

### 3. Hybrid Optimization Wins
- Don't write custom kernels for everything
- Use Triton for: memory-bound ops, fusion opportunities
- Use PyTorch/BLAS for: large matrix multiplies
- Profile to decide!

### 4. Measure Accurately
- GPU operations are asynchronous
- Always `torch.cuda.synchronize()` for accurate timing
- Without sync, timings are meaningless

### 5. Iterative Debugging
- Fix one issue at a time
- Re-measure after each fix
- Profile to identify next bottleneck
- Repeat until optimal

---

##  Files Overview

```
version3_triton/
 README_WORKSHOP.md                      # This file
 tiny_llama_v3.py                        # Main optimized model
 exercises/
    performance_debugging/              #  Hands-on debugging exercise
       README.md                       # Complete optimization journey
       run_all_stages.sh               # Run all 5 stages with profiling
       WORKSHOP_GUIDE.md               # Quick reference guide
    exercise1_triton_basics.md          # Triton fundamentals
    exercise2_swiglu_optimization.md    # SwiGLU deep dive
    exercise3_flash_attention.md        # Flash Attention implementation
 triton_profiles/                        # Generated profiling data
```

---

##  Next Steps

### After Running V3

1. **Compare with V1:**
```bash
# Run V1 for comparison
cd ../version1_pytorch_baseline/
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 20

# Compare outputs
# V1: 372.9 samp/s, 522.3 MB
# V3: 2065.0 samp/s, 281.8 MB (5.5x faster, 46% less memory!)
```

2. **Try V4 (Ultra-Fused):**
```bash
cd ../version4_pytorch_sdpa/
python tiny_llama_v4.py --batch-size 8 --seq-len 128 --num-steps 20

# Expected: ~8x faster than V1!
```

3. **Deep Dive into Profiling:**
```bash
cd exercises/performance_debugging/
./run_all_stages.sh

# Analyze the profiling CSV files
# Compare kernel execution times
# Understand the optimization journey
```

---

##  Common Issues and Solutions

### Issue 1: ImportError: No module named 'triton'
```bash
pip install triton
```

### Issue 2: RuntimeError: CUDA not available
```bash
# Verify ROCm installation
rocminfo

# Check PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue 3: Loss is not ~7.0
- Check weight initialization is enabled
- Verify model architecture matches V1
- Check for tensor shape mismatches

### Issue 4: Performance slower than expected
- Ensure tensors are contiguous: `.contiguous()`
- Check CUDA synchronization for accurate timing
- Profile to identify bottleneck kernel
- Verify using optimized SwiGLU (hybrid approach)

---

##  Additional Resources

- **Triton Documentation:** https://triton-lang.org/
- **Flash Attention Paper:** https://arxiv.org/abs/2205.14135
- **ROCm Profiling Guide:** https://rocm.docs.amd.com/projects/rocprofiler/
- **Performance Debugging Guide:** exercises/performance_debugging/README.md

---

##  Summary

**V3 achieves 5.5x speedup through:**
1.  Flash Attention (Triton) - 46% memory reduction
2.  RMSNorm (Triton) - Fused kernel
3.  Hybrid SwiGLU - Use rocBLAS for matmul
4.  Tensor contiguity - Critical for Triton performance
5.  Proper initialization - Correctness first!

**Key insight:** Best performance comes from using the right tool for each operation - not from using custom kernels everywhere!

**Ready to debug?** Start with `cd exercises/performance_debugging/`


