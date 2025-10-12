
## Exercise 2: Memory Analysis and Optimization

`exercise2_memory_analysis.md` from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline` in the Training Examples repository

### Objective
Understand memory usage patterns, identify memory bottlenecks, and analyze memory bandwidth utilization in the baseline Tiny LLaMA model.

### Prerequisites
- Completed Exercise 1
- Basic understanding of GPU memory hierarchy

### Duration
**Estimated Time:** 25-30 minutes

### Background

Memory optimization is crucial for transformer models because:
- **Memory Bandwidth**: Often the limiting factor for inference
- **Peak Memory**: Determines maximum batch size and model size
- **Memory Fragmentation**: Can reduce effective memory utilization
- **Attention Memory**: Quadratic scaling with sequence length

### Instructions

#### Step 1: Memory-Focused Profiling (10 minutes)

Run profiling with enhanced memory analysis:

```bash
## Memory profiling with different batch sizes
python tiny_llama_v1.py \
    --batch-size 4 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --enable-memory-profiling \
    --profile-dir ./memory_analysis_bs4

python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --enable-memory-profiling \
    --profile-dir ./memory_analysis_bs8

python tiny_llama_v1.py \
    --batch-size 16 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --enable-memory-profiling \
    --profile-dir ./memory_analysis_bs16
```

**📝 Record memory usage for each batch size:**

| Batch Size | Peak Memory (MB) | Avg Memory (MB) | Training Speed (samples/sec) |
|------------|------------------|-----------------|------------------------------|
| 4          |                  |                 |                              |
| 8          |                  |                 |                              |
| 16         |                  |                 |                              |

#### Step 2: Memory Timeline Analysis (10 minutes)

Analyze memory patterns using TensorBoard:

```bash
## Launch TensorBoard for memory analysis
tensorboard --logdir ./memory_analysis_bs8 --port 6007
```

In TensorBoard:
1. Go to the **PROFILE** tab
2. Select **Memory Timeline** view
3. Examine the memory usage pattern

**📝 Memory Pattern Analysis:**

**Memory Allocation Timeline:**
- At what point does memory usage peak? ________________
- What operations cause the largest memory spikes? ________________
- Are there memory deallocations visible? ________________

**Memory Efficiency:**
- Is memory usage steady or fluctuating? ________________
- Are there unnecessary memory allocations? ________________
- What's the memory utilization pattern during attention? ________________

#### Step 3: Sequence Length Scaling (8 minutes)

Test how memory scales with sequence length:

```bash
## Test different sequence lengths
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 64 \
    --num-steps 10 \
    --enable-memory-profiling \
    --profile-dir ./memory_seq64

python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 256 \
    --num-steps 10 \
    --enable-memory-profiling \
    --profile-dir ./memory_seq256

## Note: seq-len 512 might OOM - try with smaller batch size if needed
python tiny_llama_v1.py \
    --batch-size 4 \
    --seq-len 512 \
    --num-steps 5 \
    --enable-memory-profiling \
    --profile-dir ./memory_seq512
```

**📝 Sequence Length Scaling Analysis:**

| Seq Length | Batch Size | Peak Memory (MB) | Memory per Token | Scaling Pattern |
|------------|------------|------------------|------------------|-----------------|
| 64         | 8          |                  |                  |                 |
| 128        | 8          |                  |                  |                 |
| 256        | 8          |                  |                  |                 |
| 512        | 4          |                  |                  |                 |

**Memory Scaling Questions:**
1. Is memory scaling linear, quadratic, or something else with sequence length?
2. Which component shows the steepest memory scaling?
3. At what sequence length do you hit memory limits?

#### Step 4: Memory Bandwidth Analysis (7 minutes)

Use the memory profiling results to analyze bandwidth utilization:

```bash
## Run bandwidth-focused analysis
python run_deepspeed_flops.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 15 \
    --computational-intensity \
    --output-dir ./bandwidth_analysis
```

**📝 Bandwidth Analysis Results:**

Check the `bandwidth_analysis/computational_intensity.json` file:

```bash
## View bandwidth metrics
python -c "
import json
data = json.load(open('./bandwidth_analysis/computational_intensity.json'))
print('Arithmetic Intensity:', data['arithmetic_intensity_flops_per_byte'])
print('Memory Bandwidth Used:', data['memory_bandwidth_used_gb_per_sec'], 'GB/s')
print('Bandwidth Utilization:', data['memory_bandwidth_utilization_percent'], '%')
print('Workload Type:', data['memory_bound_vs_compute_bound'])
"
```

**Key Metrics:**
- Arithmetic Intensity: _______ FLOPS/byte
- Memory Bandwidth Used: _______ GB/s
- Bandwidth Utilization: _______ %
- Workload Classification: _______

### Analysis and Interpretation

#### Step 5: Memory Optimization Opportunities (10 minutes)

Based on your analysis, identify optimization opportunities:

**📝 Memory Optimization Assessment:**

**1. Memory Scaling Efficiency**
- [ ] Linear scaling with batch size (good)
- [ ] Quadratic scaling with sequence length (attention bottleneck)
- [ ] Peak memory much higher than average (fragmentation)
- [ ] Memory plateaus (good memory reuse)

**2. Bandwidth Utilization**
- [ ] High bandwidth utilization (>70%) - compute bound
- [ ] Medium bandwidth utilization (30-70%) - mixed workload
- [ ] Low bandwidth utilization (<30%) - memory bound

**3. Memory Hotspots** (check profiling results)
- [ ] Attention QKV matrices
- [ ] Attention score computation
- [ ] Feed-forward intermediate tensors
- [ ] Gradient accumulation

**4. Optimization Targets**
Rank these optimizations by memory impact (1=highest, 4=lowest):
- [ ] Flash Attention (reduce attention memory) - Rank: ___
- [ ] Gradient checkpointing (trade compute for memory) - Rank: ___
- [ ] Mixed precision (reduce memory per parameter) - Rank: ___
- [ ] Tensor fusion (reduce intermediate allocations) - Rank: ___

#### Step 6: Memory Bottleneck Identification (5 minutes)

Determine if your workload is memory-bound or compute-bound:

**📝 Bottleneck Classification:**

Based on your bandwidth analysis:
- **Arithmetic Intensity < 10 FLOPS/byte** → Memory-bound workload
- **Arithmetic Intensity 10-100 FLOPS/byte** → Mixed workload
- **Arithmetic Intensity > 100 FLOPS/byte** → Compute-bound workload

**Your Classification:** _______________________

**Evidence:**
- Arithmetic intensity: _______ FLOPS/byte
- Memory bandwidth utilization: _______ %
- GPU compute utilization: _______ % (from Exercise 1)

**Primary Bottleneck:**
- [ ] Memory bandwidth (low compute util, high memory util)
- [ ] Compute throughput (high compute util, low memory util)
- [ ] Mixed (balanced utilization)
- [ ] Kernel overhead (low both)

### Expected Results

#### Memory Usage Patterns
- **Peak Memory Growth**: Approximately linear with batch size
- **Sequence Scaling**: Quadratic scaling due to attention matrices
- **Memory Hotspots**: Attention computation and intermediate tensors
- **Bandwidth Utilization**: 30-60% on most modern GPUs

#### Key Findings
1. **Attention Memory**: Consumes significant memory, scales quadratically
2. **Memory Fragmentation**: Multiple small allocations create overhead
3. **Peak vs Average**: Large difference indicates optimization opportunity
4. **Bandwidth Bound**: Likely memory-bound for typical configurations

### Troubleshooting

**Out of Memory Errors:**
```bash
## Reduce batch size and/or sequence length
python tiny_llama_v1.py --batch-size 2 --seq-len 64
```

**Memory Profiling Failed:**
```bash
## Check CUDA memory debugging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Bandwidth Analysis Error:**
```bash
## Check DeepSpeed installation
pip install deepspeed
```

### Analysis Questions

**📝 Critical Analysis Questions:**

1. **What is the memory scaling behavior?**
   - Batch size scaling: [ ] Linear [ ] Quadratic [ ] Exponential
   - Sequence length scaling: [ ] Linear [ ] Quadratic [ ] Exponential

2. **Where is peak memory consumed?**
   - [ ] During forward pass (activations)
   - [ ] During backward pass (gradients)
   - [ ] During optimizer step (parameters)

3. **What is the primary memory optimization target?**
   - [ ] Reduce attention memory (Flash Attention)
   - [ ] Reduce activation memory (checkpointing)
   - [ ] Reduce parameter memory (mixed precision)
   - [ ] Reduce fragmentation (tensor fusion)

4. **Is the workload memory-bound or compute-bound?**
   - [ ] Memory-bound (low arithmetic intensity)
   - [ ] Compute-bound (high arithmetic intensity)
   - [ ] Mixed workload (balanced)

5. **What memory optimization would provide the biggest benefit?**
   - [ ] Flash Attention (quadratic → linear attention memory)
   - [ ] Gradient checkpointing (trade compute for memory)
   - [ ] Mixed precision FP16/BF16 (2x memory reduction)
   - [ ] Tensor fusion (reduce intermediate allocations)

### Next Steps

1. **Document your memory analysis** results
2. **Compare memory patterns** across different configurations
3. **Identify top memory optimization targets** for Version 2
4. **Understand the memory vs compute trade-offs**
5. **Proceed to Exercise 3** for bottleneck identification

### Success Criteria

**Exercise Complete When:**
- [ ] Memory profiling completed for multiple configurations
- [ ] Memory scaling patterns understood
- [ ] Bandwidth utilization analyzed
- [ ] Memory bottlenecks identified
- [ ] Optimization priorities ranked

---

**Key Takeaway**: Memory analysis reveals that the baseline model has significant memory optimization opportunities, particularly in attention computation which scales quadratically with sequence length. Flash Attention and kernel fusion will be primary targets for Version 2.

**Next Exercise**: [Exercise 3 - Bottleneck Identification](exercise_3_bottleneck_identification.md)


