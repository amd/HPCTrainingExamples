# Exercise 1: Baseline Performance Analysis

## Objective
Establish baseline performance metrics for Tiny LLaMA V1 and understand the profiling methodology that will be used throughout the workshop.

## Prerequisites
- Completed environment setup from `../setup/`
- Verified environment with validation scripts

## Duration
**Estimated Time:** 20-30 minutes

## Instructions

### Step 1: Run Baseline Training (5 minutes)

First, let's run the basic model without any profiling to establish a clean baseline:

```bash
# Navigate to version1_pytorch_baseline directory
cd version1_pytorch_baseline

# Run basic training
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 20
```

**Expected Output:**
- Model configuration summary
- Training progress with loss values
- Performance metrics (samples/sec, memory usage)
- Final performance summary

**📝 Record the following baseline metrics:**
- Training speed: _____ samples/sec
- Peak memory usage: _____ MB
- Final loss: _____
- Average batch time: _____ ms

### Step 2: Enable Basic Profiling (10 minutes)

Now let's add PyTorch profiler to understand what's happening under the hood:

```bash
# Run with PyTorch profiler enabled
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --profile-dir ./exercise1_profiles
```

**Expected Output:**
- Same training output as before
- Additional profiling information
- Profile files generated in `./exercise1_profiles/`

**📝 Answer these questions:**
1. How much overhead did profiling add to training time?
2. What files were generated in the `exercise1_profiles/` directory?
3. What's the difference in memory usage with profiling enabled?

### Step 3: Analyze Profiling Results (10 minutes)

Launch TensorBoard to visualize the profiling results:

```bash
# Launch TensorBoard (run in background)
tensorboard --logdir ./exercise1_profiles --port 6006 &

# If TensorBoard is not available, examine the JSON traces
ls -la ./exercise1_profiles/
```

**TensorBoard Analysis:**
1. Open your browser to `http://localhost:6006`
2. Navigate to the "PROFILE" tab
3. Select the most recent run

**📝 Explore and document:**

**Trace Timeline:**
- What are the top 3 longest-running operations?
  1. _________________
  2. _________________
  3. _________________

**Operator View:**
- Which operation consumes the most GPU time?
- What percentage of time is spent in attention operations?
- How many different kernel types are launched?

**Memory Timeline:**
- What is the peak memory usage?
- When does peak memory occur (forward/backward pass)?
- Are there any memory spikes or unusual patterns?

### Step 4: Identify Performance Patterns (5 minutes)

Based on your analysis, identify patterns in the baseline model:

**📝 Pattern Analysis:**

**Compute Patterns:**
- [ ] Attention operations dominate compute time
- [ ] Matrix multiplications are the primary kernels
- [ ] Many small operations with low utilization
- [ ] Memory transfers visible between operations

**Memory Patterns:**
- [ ] Memory usage grows during forward pass
- [ ] Peak memory during attention computation
- [ ] Frequent small allocations
- [ ] Memory fragmentation visible

**Optimization Opportunities:**
Based on the profiling results, which of these optimizations would likely provide the biggest benefit:
- [ ] Kernel fusion (reduce number of operations)
- [ ] Memory layout optimization
- [ ] Flash Attention implementation
- [ ] Mixed precision training
- [ ] Batch size scaling

## Expected Results

After completing this exercise, you should have:

### Performance Baseline
- **Training Speed**: 50-100 samples/sec (varies by hardware)
- **GPU Utilization**: 60-75% (typical for baseline PyTorch)
- **Memory Usage**: 2-4 GB depending on batch size
- **Kernel Count**: 40-50 different kernel launches per step

### Key Observations
- Attention operations consume ~40% of total compute time
- Matrix multiplications (GEMM) are the dominant kernels
- Multiple small operations create kernel launch overhead
- Memory allocation patterns show optimization opportunities

### Profiling Data Generated
```
exercise1_profiles/
├── events.out.tfevents.*           # TensorBoard events
├── trace_step_*.json               # Chrome trace files
├── performance_summary.json        # Performance metrics
└── [additional profile files]
```

## Troubleshooting

### Common Issues

**1. CUDA/ROCm Memory Errors**
```bash
# Reduce batch size if you get OOM errors
python tiny_llama_v1.py --batch-size 4 --seq-len 64 --num-steps 10
```

**2. Profiling Files Not Generated**
```bash
# Check permissions and disk space
ls -la ./exercise1_profiles/
df -h .
```

**3. TensorBoard Not Loading**
```bash
# Try different port or check firewall
tensorboard --logdir ./exercise1_profiles --port 6007
# Or examine JSON files directly
python -c "import json; print(json.load(open('./exercise1_profiles/performance_summary.json')))"
```

**4. Low GPU Utilization**
```bash
# Check if GPU is being used
nvidia-smi  # for NVIDIA
# or
rocm-smi   # for AMD
```

## Analysis Questions

**📝 Answer these questions based on your results:**

1. **What is the primary bottleneck in the baseline model?**
   - [ ] Memory bandwidth
   - [ ] Compute utilization
   - [ ] Kernel launch overhead
   - [ ] Data loading

2. **Which operations would benefit most from fusion?**
   - [ ] QKV projections in attention
   - [ ] Gate/Up projections in SwiGLU
   - [ ] Layer normalization operations
   - [ ] All of the above

3. **What is the Model FLOPS Utilization (rough estimate)?**
   - [ ] < 20% (memory bound)
   - [ ] 20-40% (mixed workload)
   - [ ] 40-60% (compute bound)
   - [ ] > 60% (highly optimized)

4. **Based on memory usage patterns, what optimization would help most?**
   - [ ] Gradient checkpointing
   - [ ] Flash Attention
   - [ ] Mixed precision
   - [ ] Tensor fusion

## Next Steps

After completing this exercise:

1. **Document your findings** using the performance template in the main README
2. **Compare with expected results** - are your metrics in the expected ranges?
3. **Identify top 3 optimization targets** for Version 2
4. **Proceed to Exercise 2** for memory analysis
5. **Save your profiling data** - you'll compare against Version 2 later

## Success Criteria

**Exercise Complete When:**
- [ ] Baseline training runs successfully
- [ ] Profiling data generated and analyzed
- [ ] Performance metrics documented
- [ ] Bottlenecks identified
- [ ] Ready to proceed to memory analysis

---

**Key Takeaway**: The baseline model provides a solid foundation for optimization. The profiling data clearly shows opportunities for kernel fusion, memory optimization, and attention improvements that will be addressed in subsequent versions.

**Next Exercise**: [Exercise 2 - Memory Analysis](exercise_2_memory_analysis.md)