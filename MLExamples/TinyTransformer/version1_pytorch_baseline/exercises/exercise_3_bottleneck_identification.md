
## Exercise 3: Bottleneck Identification and Optimization Planning

`exercise3_bottleneck_identification.md` from `HPCTrainingExamples/MLExamples/TinyTransformer/version1_pytorch_baseline` in the Training Examples repository

### Objective
Systematically identify performance bottlenecks in the baseline model and create an optimization roadmap for Version 2 and beyond.

### Prerequisites
- Completed Exercises 1 and 2
- Understanding of profiling results analysis

### Duration
**Estimated Time:** 30-35 minutes

### Background

Bottleneck identification is critical for effective optimization:
- **Amdahl's Law**: Overall speedup is limited by the slowest component
- **Optimization ROI**: Focus effort where it provides maximum benefit
- **Systematic Approach**: Use data-driven decisions rather than intuition
- **Baseline Establishment**: Create benchmarks for measuring improvement

### Instructions

#### Step 1: Comprehensive Profiling Run (10 minutes)

Run the complete profiling suite to gather all necessary data:

```bash
## Run comprehensive profiling analysis
bash run_all_profilers.sh \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 30 \
    --profile-dir ./bottleneck_analysis
```

This will generate:
- Baseline training metrics
- PyTorch profiler results
- FLOPS analysis data
- Memory usage patterns
- Comprehensive reports

**📝 Wait for completion and record:**
- Overall runtime: _______ seconds
- Profile data location: _______
- Any errors or warnings: _______

#### Step 2: Operator-Level Bottleneck Analysis (10 minutes)

Analyze the detailed profiling results to identify computational bottlenecks:

```bash
## View the comprehensive profiling report
cat ./bottleneck_analysis/performance_summary_report.md

## Examine PyTorch profiler operator breakdown
python run_pytorch_profiler.py \
    --analyze-existing ./bottleneck_analysis/pytorch_profiling \
    --generate-report \
    --output-dir ./detailed_analysis
```

**📝 Top Time-Consuming Operations:**

From the PyTorch profiler results, identify the top 10 operations by GPU time:

| Rank | Operation Name | GPU Time (%) | CPU Time (%) | Count | Optimization Target |
|------|----------------|-------------|-------------|-------|-------------------|
| 1    |                |             |             |       |                   |
| 2    |                |             |             |       |                   |
| 3    |                |             |             |       |                   |
| 4    |                |             |             |       |                   |
| 5    |                |             |             |       |                   |
| 6    |                |             |             |       |                   |
| 7    |                |             |             |       |                   |
| 8    |                |             |             |       |                   |
| 9    |                |             |             |       |                   |
| 10   |                |             |             |       |                   |

**Pattern Analysis:**
- What percentage of time is spent in matrix multiplications? _______%
- How many separate linear projection operations are there? _______
- What's the overhead from kernel launches vs. computation? _______%

#### Step 3: FLOPS Efficiency Analysis (8 minutes)

Examine computational efficiency using the FLOPS analysis:

```bash
## View FLOPS analysis results
python -c "
import json
with open('./bottleneck_analysis/flops_analysis/flops_profile.json', 'r') as f:
    data = json.load(f)

print('=== FLOPS EFFICIENCY ANALYSIS ===')
print(f'Model FLOPS Utilization: {data[\"efficiency_metrics\"][\"mfu_percent\"]:.1f}%')
print(f'Achieved FLOPS/sec: {data[\"performance_metrics\"][\"flops_per_sec\"]:.2e}')
print(f'Peak Device FLOPS: {data[\"efficiency_metrics\"][\"device_peak_flops\"]:.2e}')
print(f'FLOPS per Parameter: {data[\"flops_analysis\"][\"flops_per_parameter\"]:.2f}')
print(f'Throughput: {data[\"performance_metrics\"][\"throughput_samples_per_sec\"]:.1f} samples/sec')
"
```

**📝 Efficiency Metrics:**
- Model FLOPS Utilization (MFU): _______%
- Achieved FLOPS per second: _______
- FLOPS per parameter: _______
- Overall throughput: _______ samples/sec

**Efficiency Classification:**
- [ ] < 20% MFU: Severely underutilized (kernel overhead dominant)
- [ ] 20-40% MFU: Memory-bound workload
- [ ] 40-60% MFU: Mixed workload with optimization opportunities
- [ ] > 60% MFU: Well-optimized compute-bound workload

#### Step 4: Memory Bottleneck Assessment (7 minutes)

Analyze memory-related bottlenecks:

```bash
## Check computational intensity analysis
python -c "
import json
import os

intensity_file = './bottleneck_analysis/flops_analysis/computational_intensity.json'
if os.path.exists(intensity_file):
    with open(intensity_file, 'r') as f:
        data = json.load(f)

    print('=== MEMORY BOTTLENECK ANALYSIS ===')
    print(f'Arithmetic Intensity: {data[\"arithmetic_intensity_flops_per_byte\"]:.2f} FLOPS/byte')
    print(f'Memory Bandwidth Used: {data[\"memory_bandwidth_used_gb_per_sec\"]:.1f} GB/s')
    print(f'Bandwidth Utilization: {data[\"memory_bandwidth_utilization_percent\"]:.1f}%')
    print(f'Workload Type: {data[\"memory_bound_vs_compute_bound\"]}')
else:
    print('Computational intensity analysis not available')
"
```

**📝 Memory Analysis:**
- Arithmetic Intensity: _______ FLOPS/byte
- Memory Bandwidth Utilization: _______%
- Primary Bottleneck: [ ] Memory-bound [ ] Compute-bound [ ] Mixed
- Peak Memory Usage: _______ MB

**Roofline Model Position:**
- [ ] Below roofline - memory bound (optimize data movement)
- [ ] On roofline - balanced (optimize both)
- [ ] Below compute ceiling - compute bound (optimize kernels)

#### Step 5: Systematic Bottleneck Ranking (10 minutes)

Create a systematic ranking of bottlenecks based on impact and effort:

**📝 Bottleneck Impact Assessment:**

For each major bottleneck, assess:

| Bottleneck Category | % of Total Time | Optimization Difficulty | Expected Speedup | Priority Rank |
|--------------------|-----------------|------------------------|------------------|---------------|
| QKV Projections    |                 | Low-Medium             | 1.2-1.5x         |               |
| Attention Computation |             | Medium                 | 1.3-2.0x         |               |
| SwiGLU Gate/Up     |                 | Low                    | 1.1-1.3x         |               |
| Kernel Launch Overhead |            | Medium-High            | 1.5-3.0x         |               |
| Memory Fragmentation |              | Medium                 | 1.1-1.4x         |               |
| Softmax Operations |                 | Medium-High            | 1.2-1.8x         |               |

**Impact vs Effort Matrix:**

High Impact, Low Effort (Priority 1):
- _______________________________
- _______________________________

High Impact, High Effort (Priority 2):
- _______________________________
- _______________________________

Low Impact, Low Effort (Priority 3):
- _______________________________
- _______________________________

Low Impact, High Effort (Priority 4 - Skip):
- _______________________________
- _______________________________

### Analysis and Optimization Roadmap

#### Step 6: Create Version 2 Optimization Plan (10 minutes)

Based on your analysis, create a detailed optimization plan for Version 2:

**📝 Version 2 Optimization Roadmap:**

**Phase 1: Kernel Fusion (Expected: 1.4-1.8x speedup)**
- [ ] **QKV Fusion**: Combine Q, K, V linear projections
  - Impact: Reduce 3 kernel launches to 1
  - Memory: Reduce intermediate tensor allocations
  - Implementation: Fused linear layer

- [ ] **SwiGLU Fusion**: Combine gate and up projections
  - Impact: Reduce 2 kernel launches to 1
  - Memory: Eliminate intermediate activations
  - Implementation: Custom fused activation

**Phase 2: Attention Optimization (Expected: 1.3-2.0x speedup)**
- [ ] **Flash Attention**: Memory-efficient attention computation
  - Impact: Reduce attention memory from O(n^2) to O(n)
  - Memory: Enable longer sequences and larger batches
  - Implementation: torch.nn.functional.scaled_dot_product_attention

**Phase 3: Additional Optimizations (Expected: 1.1-1.3x speedup)**
- [ ] **Torch Compile**: Automatic kernel fusion
- [ ] **Memory Layout**: Optimize tensor layouts
- [ ] **Mixed Precision**: FP16/BF16 where appropriate

**Expected Overall Speedup for Version 2:** _______x

#### Step 7: Validation Metrics Definition (5 minutes)

Define metrics to validate Version 2 improvements:

**📝 Success Metrics for Version 2:**

**Performance Targets:**
- Training throughput: _______ samples/sec → _______ samples/sec
- Model FLOPS Utilization: _______ % → _______ %
- Peak memory usage: _______ MB → _______ MB
- Kernel count per step: _______ → _______

**Validation Tests:**
- [ ] Batch size 8, sequence length 128 (baseline comparison)
- [ ] Batch size 16, sequence length 256 (scaling test)
- [ ] Memory scaling with sequence length
- [ ] Numerical accuracy validation (loss convergence)

**Quality Gates:**
- [ ] No degradation in model accuracy
- [ ] Deterministic execution maintained
- [ ] Memory usage reduced or stable
- [ ] Throughput improved by >30%

### Expected Results

#### Typical Bottleneck Hierarchy
1. **Attention Operations (35-45% of time)**
   - Multiple QKV projections
   - Attention score computation
   - Softmax operations

2. **Feed-Forward Network (25-35% of time)**
   - Gate and up projections
   - SwiLU activation
   - Down projection

3. **Kernel Launch Overhead (10-20% of time)**
   - Multiple small operations
   - Memory transfers between kernels

4. **Memory Operations (5-15% of time)**
   - Tensor allocations/deallocations
   - Memory fragmentation

#### Optimization Priority Order
1. **QKV Fusion** (Low effort, medium impact)
2. **Flash Attention** (Medium effort, high impact)
3. **SwiGLU Fusion** (Low effort, low-medium impact)
4. **Torch Compile** (Very low effort, variable impact)

### Troubleshooting

**Missing Analysis Files:**
```bash
## Re-run comprehensive profiling if files are missing
bash run_all_profilers.sh --batch-size 8 --profile-dir ./bottleneck_retry
```

**Profiling Data Errors:**
```bash
## Check for GPU memory issues
nvidia-smi  # or rocm-smi
## Reduce batch size if necessary
```

### Analysis Questions

**📝 Critical Analysis Questions:**

1. **What is the single largest performance bottleneck?**
   - [ ] QKV projection operations
   - [ ] Attention score computation
   - [ ] Feed-forward network
   - [ ] Kernel launch overhead
   - [ ] Memory bandwidth

2. **What type of optimization would provide the biggest benefit?**
   - [ ] Kernel fusion (reduce launches)
   - [ ] Memory optimization (bandwidth)
   - [ ] Algorithmic optimization (attention)
   - [ ] Precision optimization (mixed precision)

3. **Is the workload primarily:**
   - [ ] Memory-bound (optimize data movement)
   - [ ] Compute-bound (optimize kernels)
   - [ ] Overhead-bound (optimize launches)
   - [ ] Mixed workload (balanced optimization)

4. **What should be the first optimization implemented?**
   - [ ] QKV fusion (immediate benefit)
   - [ ] Flash Attention (biggest impact)
   - [ ] SwiGLU fusion (easy implementation)
   - [ ] Torch compile (automatic optimization)

5. **What is the realistic speedup target for Version 2?**
   - [ ] 1.2-1.4x (conservative)
   - [ ] 1.5-2.0x (achievable)
   - [ ] 2.0-3.0x (optimistic)
   - [ ] >3.0x (unlikely without major changes)

### Deliverables

At the end of this exercise, you should have:

1. **Bottleneck Analysis Report** with quantified performance issues
2. **Optimization Roadmap** with prioritized improvements
3. **Version 2 Implementation Plan** with expected benefits
4. **Success Metrics** for validating improvements
5. **Baseline Measurements** for comparison

### Next Steps

1. **Document all findings** in the performance summary template
2. **Review optimization priorities** with team/instructor
3. **Validate technical feasibility** of planned optimizations
4. **Proceed to Version 2** implementation with clear targets
5. **Set up regression testing** framework for validation

### Success Criteria

**Exercise Complete When:**
- [ ] Comprehensive bottleneck analysis completed
- [ ] Performance bottlenecks quantified and ranked
- [ ] Optimization roadmap created with priorities
- [ ] Success metrics defined for Version 2
- [ ] Implementation plan validated
- [ ] Ready to begin Version 2 optimizations

---

**Key Takeaway**: Systematic bottleneck identification reveals that the baseline model has clear optimization opportunities in kernel fusion, attention computation, and memory usage. The data-driven approach provides a roadmap for achieving 1.5-2.0x speedup in Version 2.

**Next Phase**: [Version 2 - PyTorch Fused](../version2_pytorch_fused/README.md)


