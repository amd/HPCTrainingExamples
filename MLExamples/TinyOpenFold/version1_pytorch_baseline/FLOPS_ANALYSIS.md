# DeepSpeed FLOPS Analysis for TinyOpenFold

Analyze computational efficiency and FLOPS breakdown of the Evoformer architecture using DeepSpeed profiling tools.

## Quick Start

```bash
# Basic FLOPS analysis
./run_deepspeed_flops.sh

# Comprehensive analysis with all features
./run_deepspeed_flops.sh --all

# Custom configuration
./run_deepspeed_flops.sh --batch-size 8 --seq-len 128 --num-blocks 8

# Install DeepSpeed if needed
pip install deepspeed
```

## What You Get

The FLOPS profiler provides:
- **Total FLOPS** per training step
- **FLOPS breakdown** by component (MSA attention, triangle multiplication, etc.)
- **Model FLOPS Utilization (MFU)** - GPU efficiency metric
- **Computational intensity** - memory vs compute bound classification
- **Roofline model data** - optimization recommendations

**Example Output:**
```
FLOPS Analysis Summary:
   Total FLOPS per step: 2.45e+11
   Model FLOPS Utilization: 15.3%
   
Evoformer FLOPS Breakdown:
   msa_attention: 8.32e+10 (34.0%)
   triangle_multiplication: 6.21e+10 (25.4%)
   pair_transition: 4.15e+10 (17.0%)
```

## Key Metrics

### Model FLOPS Utilization (MFU)

```
MFU = (Achieved FLOPS) / (Peak GPU FLOPS) × 100%
```

**Targets:**
- < 20%: Heavy overhead, needs kernel fusion
- 20-40%: Typical unoptimized baseline
- 40-60%: Good optimization
- 60-80%: Excellent (state-of-the-art)

### Computational Intensity

```bash
./run_deepspeed_flops.sh --intensity
```

**Classification:**
- < 10 FLOPS/byte: Memory-bound
- 10-50 FLOPS/byte: Balanced
- \> 50 FLOPS/byte: Compute-bound

## Common Commands

```bash
# Identify bottlenecks
./run_deepspeed_flops.sh --all --output-dir analysis
cat analysis/flops_profile.json | jq '.flops_analysis.evoformer_breakdown'

# Multi-GPU analysis
./run_deepspeed_flops.sh --multi-gpu --output-dir multi_gpu_results

# Specific GPUs
./run_deepspeed_flops.sh --devices "0,1,2,3"

# Roofline analysis
./run_deepspeed_flops.sh --roofline --output-dir roofline_data
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size <n>` | Batch size | 4 |
| `--seq-len <n>` | Sequence length | 64 |
| `--num-blocks <n>` | Evoformer blocks | 4 |
| `--device <n>` | GPU device ID | default |
| `--multi-gpu` | Profile all GPUs | false |
| `--devices <ids>` | Specific GPUs (e.g., "0,1,2") | none |
| `--all` | All analysis types | false |
| `--roofline` | Roofline analysis | false |
| `--intensity` | Computational intensity | false |

## Output Files

- `flops_profile.json` - Complete FLOPS analysis and efficiency metrics
- `computational_intensity.json` - Memory bandwidth analysis
- `roofline_data.json` - Roofline model data

## Optimization Priorities

Based on FLOPS breakdown:

1. **Triangle Multiplication > 25%**: Implement fused kernels (30-40% improvement)
2. **MSA Attention > 30%**: Use Flash Attention (2-3x speedup)
3. **Low MFU (< 20%)**: Apply kernel fusion, reduce Python overhead
4. **Memory-bound (AI < 10)**: Use mixed precision, optimize memory access

## GPU Specifications

| GPU | Peak FP32 TFLOPS | Memory Bandwidth | Target MFU |
|-----|------------------|------------------|------------|
| AMD MI300X | 163.4 | 5300 GB/s | 40-60% |
| NVIDIA H100 | 67 | 3350 GB/s | 45-65% |
| NVIDIA A100 | 19.5 | 2039 GB/s | 35-55% |

## References

- [DeepSpeed FLOPS Profiler](https://www.deepspeed.ai/tutorials/flops-profiler/)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
- Main documentation: `README.md`
