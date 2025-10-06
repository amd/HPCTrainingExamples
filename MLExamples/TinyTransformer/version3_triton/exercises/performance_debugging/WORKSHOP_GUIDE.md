# V3 Performance Debugging Workshop Guide

## Quick Start

```bash
cd /workspace/version3_triton/exercises/performance_debugging

# Read the comprehensive guide
cat README.md

# Note: Individual stage files (v3_stage1_broken_loss.py, etc.) are symbolic links
# to the main tiny_llama_v3.py with modifications applied at runtime or via
# configuration flags. This keeps the exercise files manageable.

# Run all stages with automatic profiling and comparison
./run_all_stages.sh

# Results will be saved to results/ directory with:
# - stage*_output.log: Full training outputs
# - stage*_profile.csv: rocprof profiling data
# - Performance comparison summary
```

## What This Exercise Teaches

This is a **realistic performance debugging scenario** that mirrors real-world optimization work:

### 1. **Correctness Before Performance** (Stage 1)
- Shows how subtle bugs (missing weight init) can completely break training
- Demonstrates diagnostic logging techniques
- Loss goes from 942 → 7.0 after one-line fix

### 2. **Memory Layout Matters** (Stage 2→3)
- Non-contiguous tensors after `repeat_interleave` killed performance
- Adding `.contiguous()` gave 20x speedup (15 → 310 samples/sec)
- Critical lesson for GPU kernel developers

### 3. **Measure Accurately** (Stage 3→4)
- GPU operations are asynchronous
- Without `torch.cuda.synchronize()`, timings are meaningless
- Same performance, but now we can see WHERE the time is spent

### 4. **Know When NOT to Use Custom Kernels** (Stage 4→5)
- Triton SwiGLU kernel was launching 2M+ threads
- Each doing naive matrix multiplication
- PyTorch's rocBLAS is orders of magnitude faster
- Result: 8x forward pass speedup (25.5ms → 3.2ms)

### 5. **Hybrid Optimization Wins**
- Final version: 2065 samples/sec (5.5x faster than V1 baseline!)
- Uses Triton for: Flash Attention, RMSNorm (memory-bound ops)
- Uses PyTorch for: Matrix multiplies (compute-bound ops)
- **Best of both worlds**

## For Workshop Participants

### Beginner Level
1. Run `./run_all_stages.sh` and observe the progression
2. Read the output logs to understand what changed each stage
3. Focus on the "Key Observations" in the comparison summary

### Intermediate Level
1. Examine the profiling CSV files in `results/`
2. Compare kernel execution times between stages
3. Try modifying block sizes in Flash Attention kernel
4. Re-run and observe impact on performance

### Advanced Level
1. Use `rocprofv2 --kernel-trace` for detailed timeline analysis
2. Identify memory bandwidth bottlenecks
3. Experiment with different Triton kernel implementations
4. Write a custom kernel for RoPE application

## Key Takeaways

| Metric | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|--------|---------|---------|---------|---------|---------|
| **Loss** |  942 |  7.0 |  7.0 |  7.0 |  7.0 |
| **Speed** | N/A | 15 samp/s | 311 samp/s | 306 samp/s | **2065 samp/s** |
| **vs Baseline** | N/A | 0.04x | 0.83x | 0.82x | **5.5x** |
| **Key Issue** | No weight init | Non-contig tensors | Wrong timing | Slow SwiGLU | **OPTIMAL** |
| **Memory** | N/A | ~282 MB | ~282 MB | ~282 MB | **~282 MB** |

**Baseline (V1):** 372.9 samples/sec, 522.3 MB

## Profiling Commands Reference

```bash
# Basic profiling with rocprof
rocprof --stats python tiny_llama_v3.py --batch-size 8 --seq-len 128 --num-steps 20

# Detailed kernel trace
rocprofv2 --kernel-trace -o trace.json python tiny_llama_v3.py ...

# View trace in Perfetto
# Upload trace.json to https://ui.perfetto.dev

# Compare two stages
diff results/stage2_profile.csv results/stage5_profile.csv

# Find slowest kernels
sort -t',' -k4 -nr results/stage4_profile.csv | head -20
```

## Common Questions

**Q: Why not just use the final optimized version?**
A: Understanding the journey is more valuable than the destination. Each stage teaches a critical lesson about GPU programming and performance debugging.

**Q: Can I apply these techniques to my own models?**
A: Absolutely! The debugging methodology is universal:
   1. Ensure correctness first
   2. Add accurate timing/profiling
   3. Identify bottlenecks with profilers
   4. Fix one issue at a time
   5. Re-measure and validate

**Q: Should I always use Triton for custom kernels?**
A: No! As Stage 5 shows, hybrid approaches work best:
   - Use Triton for memory-bound, fusion opportunities (Flash Attention, layer norm)
   - Use PyTorch/BLAS for compute-bound matrix ops
   - Profile to verify your assumptions

**Q: Why is memory usage the same across all stages?**
A: The memory footprint is determined by model architecture (activations, weights, gradients), not by the kernel implementations. The performance gains come from faster computation, not lower memory usage. Flash Attention provides memory savings by avoiding materialization of the full attention matrix.

## Next Steps

After completing this exercise:

1. **Apply to V4**: The ultra-fused version has similar issues - try fixing them yourself
2. **Explore ROCm Tools**: Deep dive into rocprofv2, rocprof, omniperf
3. **Custom Kernels**: Write your own Triton kernel for a simple operation
4. **Production Deployment**: Consider trade-offs between development time and performance gains

## Additional Resources

- **Triton Tutorials**: https://triton-lang.org/main/getting-started/tutorials/index.html
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **ROCm Profiling**: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
- **PyTorch Profiler**: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

---

**Exercise Created**: October 2025
**Target Hardware**: AMD MI325X with ROCm 6.4.4
**Framework**: PyTorch 2.7.1 + Triton
