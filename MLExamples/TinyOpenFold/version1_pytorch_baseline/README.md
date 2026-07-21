# TinyOpenFold V1: PyTorch Baseline

Educational implementation of AlphaFold 2's Evoformer architecture with comprehensive profiling integration.

## Overview

This version provides a clean, well-documented baseline implementation of the core AlphaFold 2 architecture, focusing on the **Evoformer** blocks that process MSA (Multiple Sequence Alignment) and pair representations.

## Quick Start

### Basic Training Run

```bash
# Default configuration: 64 residues, 16 MSA sequences, 4 Evoformer blocks
python tiny_openfold_v1.py --batch-size 4 --num-steps 30

# Expected output:
# Model Configuration:
#    MSA dimension: 64
#    Pair dimension: 128
#    Evoformer blocks: 4
#    Total parameters: 2,641,728
#    Model size: 10.6 MB (FP32)
#
# Training steps complete with loss decreasing
```

### Validation Check

```bash
# Verify your environment is set up correctly
python tiny_openfold_v1.py --validate-setup

# Should output:
# Validation successful! Environment ready.
```

## Architecture Components

### 1. MSA Representation Processing

**MSA Row-wise Attention with Pair Bias**
- Attends across residues within each MSA sequence
- Biased by the pair representation (key innovation!)
- Shape: `(batch, n_seqs, seq_len, msa_dim)`

**MSA Column-wise Attention**
- Attends across different sequences for each position
- Enables communication between sequences in the MSA
- Shape: `(batch, n_seqs, seq_len, msa_dim)`

**MSA Transition**
- Point-wise feed-forward network
- Applied to each MSA element independently

### 2. Pair Representation Processing

**Outer Product Mean**
- Projects MSA patterns onto pairwise space
- Computes mean outer product across MSA sequences
- Updates pair representation with sequence information

**Triangle Multiplicative Updates**
- Geometric reasoning: if i-j and j-k are close, i-k should be considered
- Two versions: outgoing and incoming edges
- Most computationally expensive operation (O(N³))

**Triangle Self-Attention**
- Attention over edges in the residue graph
- Two versions: starting and ending nodes
- Enables long-range communication

**Pair Transition**
- Point-wise feed-forward network for pair representation

### 3. Structure Module

**Simplified Distance Prediction**
- Predicts pairwise distances from pair representation
- In full AlphaFold 2, this is the Invariant Point Attention (IPA) module
- Output: `(batch, seq_len, seq_len, 1)` - distance matrix

## Model Configuration

### Default Configuration

```python
TinyOpenFoldConfig(
    vocab_size=21,              # 20 amino acids + unknown
    msa_dim=64,                 # MSA feature dimension
    pair_dim=128,               # Pair feature dimension
    n_evoformer_blocks=4,       # Number of Evoformer blocks
    n_heads_msa=4,              # MSA attention heads
    n_heads_pair=4,             # Pair attention heads
    msa_intermediate_dim=256,   # MSA FFN dimension (4x msa_dim)
    pair_intermediate_dim=512,  # Pair FFN dimension (4x pair_dim)
    outer_product_dim=32,       # Outer product projection dim
    max_seq_len=64,             # Maximum sequence length
    n_seqs=16,                  # Number of MSA sequences
)
```

### Scaling Configurations

#### Tiny (for testing)
```bash
python tiny_openfold_v1.py \
    --msa-dim 32 \
    --pair-dim 64 \
    --num-blocks 2 \
    --seq-len 32 \
    --num-seqs 8 \
    --batch-size 8

# Parameters: ~660K
# Memory: ~40 MB
# Speed: ~15-20 samples/sec
```

#### Small (default)
```bash
python tiny_openfold_v1.py \
    --msa-dim 64 \
    --pair-dim 128 \
    --num-blocks 4 \
    --seq-len 64 \
    --num-seqs 16 \
    --batch-size 4

# Parameters: ~2.6M
# Memory: ~100 MB
# Speed: ~8-10 samples/sec
```

#### Medium
```bash
python tiny_openfold_v1.py \
    --msa-dim 128 \
    --pair-dim 256 \
    --num-blocks 8 \
    --seq-len 128 \
    --num-seqs 32 \
    --batch-size 2

# Parameters: ~42M
# Memory: ~800 MB
# Speed: ~1-2 samples/sec
```

## Profiling Guide

### PyTorch Profiler

Detailed kernel-level performance and memory analysis:

```bash
# Basic profiling
python tiny_openfold_v1.py \
    --enable-pytorch-profiler \
    --profile-dir ./profiles \
    --batch-size 4 \
    --num-steps 30

# View timeline in Chrome
# Open chrome://tracing and load ./profiles/trace_*.json
```

**Provides:**
- Kernel execution times
- Memory allocation patterns
- CPU/GPU timeline

#### Minimal Overhead Profiling (Recommended for Throughput Measurement)

For production-like performance measurements with minimal profiling overhead:

```bash
# Default: Profile only 5 out of 20 steps (25% overhead)
./run_pytorch_profiler.sh

# Minimal overhead: Profile 5 out of 100 steps (~5% overhead)
./run_pytorch_profiler.sh \
    --batch-size 4 \
    --seq-len 64 \
    --num-steps 100 \
    --profile-steps 5 \
    --device 0

# Very stable throughput: Profile 5 out of 200 steps (~2.5% overhead)
./run_pytorch_profiler.sh \
    --num-steps 200 \
    --profile-steps 5

# View comprehensive report
less pytorch_profiles/comprehensive_profiling_report.md

# View trace in Chrome
# Open chrome://tracing and load: pytorch_profiles/trace_step_*.json
```

**Key Parameters for Minimal Overhead:**
- `--num-steps 100-200`: More steps = more stable throughput average
- `--profile-steps 5`: Only these steps have profiling overhead (~40% slower)
- Non-profiled steps: **No overhead** (82 samples/sec baseline)
- Result: Average throughput with only 5-10% overhead

**What You Get:**
- `trace_step_*.json` - Chrome trace file (~80-100 MB) for detailed kernel inspection
- `comprehensive_profiling_report.md` - Analysis with bottleneck identification
- `operator_analysis.json` - Performance data
- Throughput summary at end of comprehensive report

**Example Output:**
```
Average training speed: 75.0 samples/sec  (vs 82 baseline, 10% overhead with 5/100 profiled)
```

### DeepSpeed FLOPS Profiler

Analyze computational efficiency and FLOPS breakdown using DeepSpeed:

```bash
# Basic FLOPS analysis (single GPU, default device)
./run_deepspeed_flops.sh

# Profile on specific GPU
./run_deepspeed_flops.sh --device 1

# Multi-GPU comparative analysis (all available GPUs - 8 on MI250X)
./run_deepspeed_flops.sh --multi-gpu

# Multi-GPU analysis (specific GPUs)
./run_deepspeed_flops.sh --devices "0,1,2"

# Comprehensive analysis with roofline model
./run_deepspeed_flops.sh --all --batch-size 4 --seq-len 64

# Custom configuration
./run_deepspeed_flops.sh \
    --batch-size 8 \
    --seq-len 128 \
    --num-blocks 8 \
    --roofline \
    --intensity
```

**Key Metrics from FLOPS Analysis:**
- **Model FLOPS Utilization (MFU)**: Efficiency of GPU usage (target: 40-60% for baseline)
- **FLOPS Breakdown**: Which Evoformer components use most compute
- **Arithmetic Intensity**: Memory-bound vs compute-bound classification
- **Roofline Data**: Optimization recommendations
- **Multi-GPU Efficiency**: Scaling efficiency across multiple GPUs (target: >90% for good scaling)

**Example Output (Single GPU):**
```
FLOPS Analysis Summary:
   Total FLOPS per step: 2.45e+11
   Model FLOPS Utilization: 15.3%
   
Evoformer FLOPS Breakdown:
   msa_attention: 8.32e+10 (34.0%)
   triangle_multiplication: 6.21e+10 (25.4%)
   pair_transition: 4.15e+10 (17.0%)
```

**Example Output (Multi-GPU):**
```
Aggregate Multi-GPU Summary:
   Number of GPUs: 8
   Total System TFLOPS: 196.8
   Average MFU: 15.8%
   Total Throughput: 84.6 samples/sec
   Multi-GPU Efficiency: 95.2%
   Speedup vs Single GPU: 7.62x
```

**Multi-GPU Analysis:**
- Profiles each GPU independently to measure per-GPU FLOPS
- Calculates aggregate system TFLOPS (sum across all GPUs)
- Reports multi-GPU efficiency (actual speedup / ideal speedup)
- Identifies GPU-to-GPU performance variance (MFU std dev)
- Useful for understanding scaling bottlenecks and load balancing

**See Also:** 
- `FLOPS_ANALYSIS.md` for detailed documentation and workflows
- `PROFILER_COMPARISON_GUIDE.md` for DeepSpeed FLOPS vs PyTorch Profiler comparison

### Memory Profiling

Track memory usage throughout training:

```bash
python tiny_openfold_v1.py \
    --enable-memory-profiling \
    --profile-dir ./memory_analysis \
    --batch-size 4

# Check performance_summary.json for memory statistics
cat ./memory_analysis/performance_summary.json
```

### Complete Profiling Suite

Enable all profiling features:

```bash
python tiny_openfold_v1.py \
    --enable-all-profiling \
    --profile-dir ./complete_analysis \
    --batch-size 4 \
    --num-steps 50
```

## Performance Analysis

### Expected Bottlenecks

Based on the architecture, expect these components to dominate compute time:

1. **Triangle Operations** (40-50% of time)
   - O(N³) complexity makes these expensive
   - Both multiplicative updates and attention
   - Most sensitive to sequence length

2. **MSA Attention** (25-35% of time)
   - Row-wise attention: O(N_seqs × N_res²)
   - Column-wise attention: O(N_res × N_seqs²)
   - Depends on both MSA depth and sequence length

3. **Outer Product Mean** (10-15% of time)
   - Computing outer products across MSA
   - Memory-bound operation

4. **Transitions** (5-10% of time)
   - Feed-forward networks
   - Usually well-optimized by PyTorch

### Memory Consumption

Memory usage breakdown (approximate):

```
Total GPU Memory = Model Parameters + Activations + Gradients + Optimizer States

For batch=4, seq_len=64, n_seqs=16:
- Model: ~11 MB (FP32)
- MSA activations: ~4 MB
- Pair activations: ~32 MB
- Attention scores: ~8 MB
- Gradients: ~11 MB
- Optimizer (Adam): ~22 MB
- Total: ~90-100 MB
```

**Key Insight**: Pair representation dominates memory (seq_len²)

### Optimization Opportunities

From the baseline implementation, potential optimizations include:

1. **Flash Attention** for MSA attention operations
2. **Kernel Fusion** for triangle operations
3. **Mixed Precision (FP16)** to reduce memory and improve throughput
4. **Gradient Checkpointing** for larger models
5. **Custom CUDA/Triton Kernels** for triangle updates

## Training Output Explanation

### During Training

```
Step   0/50 | Loss: 45.2341 | Speed:   8.5 samples/sec | Memory:  102.3 MB | Time:  470.2ms
```

- **Loss**: MSE on predicted distances (should decrease)
- **Speed**: Throughput in samples/second
- **Memory**: Current GPU memory allocation
- **Time**: Milliseconds per training iteration

### Final Summary

```
Performance Summary:
   Total samples processed: 200
   Average training speed: 8.7 samples/sec
   Average batch time: 459.3 ms
   Average forward time: 285.1 ms
   Average backward time: 165.7 ms
   Average optimizer time: 8.5 ms
   Final loss: 38.4512
   Peak memory usage: 102.3 MB
```

**What to Analyze:**
- Forward/backward time ratio (typically 1.5-2.0x)
- Memory growth over time
- Loss convergence behavior

## Multi-GPU Training and Scaling Studies

### Multi-GPU Training with DataParallel

TinyOpenFold supports multi-GPU training using PyTorch's `nn.DataParallel`. The implementation automatically detects and uses multiple GPUs based on environment variables.

**Single GPU (Explicit):**
```bash
# Use specific GPU
python tiny_openfold_v1.py --device 0 --batch-size 8
```

**Multi-GPU (Automatic Detection):**
```bash
# ROCm (AMD GPUs) - automatically uses GPUs 0 and 1
ROCR_VISIBLE_DEVICES=0,1 python tiny_openfold_v1.py --batch-size 16

# CUDA (NVIDIA GPUs) - automatically uses GPUs 0, 1, 2, 3
CUDA_VISIBLE_DEVICES=0,1,2,3 python tiny_openfold_v1.py --batch-size 32

# Disable multi-GPU even if multiple GPUs are available
python tiny_openfold_v1.py --no-data-parallel --device 0 --batch-size 8
```

**Best Practices:**
- Scale batch size proportionally with GPU count (e.g., 8 per GPU)
- The effective batch size is split across GPUs automatically
- Monitor per-GPU memory usage to avoid OOM errors
- Use `--device` to override automatic GPU detection for single-GPU runs

### Running Scaling Studies

Two scripts are provided for conducting GPU scaling studies:

#### Quick Scaling Test (Simple)

For a quick test with 1, 2, 4, and 8 GPUs:

```bash
# Make script executable
chmod +x quick_scaling_test.sh

# Run quick scaling test (8 samples per GPU, 50 steps)
./quick_scaling_test.sh
```

**Output:**
- Creates timestamped directory with logs for each GPU configuration
- Automatically calculates speedup and efficiency
- Generates summary table with throughput comparison

**Example Results:**
```
GPUs     Throughput (s/s)     Speedup      Efficiency
----     -------------------  ---------    ----------
1        166.9                1.00x        100.0%
2        202.7                1.21x        60.5%
4        245.3                1.47x        36.8%
8        249.1                1.49x        18.6%
```

#### Comprehensive Scaling Study (Advanced)

For more control and statistical analysis:

```bash
# Make script executable
chmod +x run.sh

# Run full scaling study with defaults
./run.sh

# Custom configuration
./run.sh --gpus "1 2 4 8" --batch-per-gpu 8 --steps 100 --runs 3

# With mixed precision and profiling
./run.sh --amp --profile --steps 50

# Specify output directory
./run.sh --output-dir my_scaling_study_$(date +%Y%m%d)

# Show help
./run.sh --help
```

**Options:**
- `--gpus <list>`: GPU counts to test (default: "1 2 4 8")
- `--batch-per-gpu <n>`: Batch size per GPU (default: 8)
- `--steps <n>`: Training steps per run (default: 50)
- `--runs <n>`: Number of runs per configuration for statistics (default: 1)
- `--amp`: Enable mixed precision training (FP16)
- `--profile`: Enable PyTorch profiler
- `--output-dir <dir>`: Custom output directory

**Output Files:**
```
scaling_study_TIMESTAMP/
├── config.txt                    # Study configuration
├── summary.txt                   # Human-readable summary with statistics
├── summary.csv                   # Machine-readable results
├── gpu1_batch8_run1.log          # Detailed logs for each run
├── gpu2_batch16_run1.log
├── gpu4_batch32_run1.log
└── gpu8_batch64_run1.log
```

### Understanding Scaling Efficiency

**Scaling Metrics:**
- **Speedup**: `Throughput(N GPUs) / Throughput(1 GPU)`
- **Efficiency**: `(Speedup / N GPUs) × 100%`

**Expected Behavior:**
- **Ideal Linear Scaling**: 100% efficiency (rare in practice)
- **Good Scaling**: 70-90% efficiency for 2-4 GPUs
- **Diminishing Returns**: Efficiency drops with more GPUs due to:
  - Communication overhead between GPUs
  - DataParallel synchronization costs
  - Small model size (2.6M parameters)
  - Memory bandwidth limitations

**TinyOpenFold Scaling Characteristics:**
- Sub-linear scaling is expected due to small model size
- Communication overhead becomes significant at 4+ GPUs
- Best efficiency typically at 2-4 GPUs
- Beyond 8 GPUs, overhead may exceed benefits for this model size

**Optimization Tips:**
- Use larger batch sizes per GPU to amortize communication costs
- Enable mixed precision (`--use-amp`) to reduce memory and increase throughput
- Consider gradient accumulation for effective larger batch sizes
- For production OpenFold, use model parallelism instead of data parallelism

## Command Reference

### Model Configuration
```bash
--msa-dim 64              # MSA representation dimension
--pair-dim 128            # Pair representation dimension  
--num-blocks 4            # Number of Evoformer blocks
--num-seqs 16             # Number of MSA sequences
--seq-len 64              # Sequence length (residues)
```

### Training Parameters
```bash
--num-steps 50            # Training iterations
--batch-size 4            # Batch size
--learning-rate 3e-4      # Learning rate
--use-amp                 # Enable mixed precision (FP16)
```

### Profiling Options
```bash
--enable-pytorch-profiler # Enable PyTorch profiler
--enable-memory-profiling # Track memory usage
--enable-all-profiling    # Enable all profiling
--profile-dir PATH        # Output directory
--warmup-steps 3          # Profiler warmup iterations
--profile-steps 5         # Iterations to profile
```

## Code Structure

### Main Classes

**`TinyOpenFoldConfig`**: Model configuration dataclass

**`MSARowAttentionWithPairBias`**: MSA row attention + pair bias
- Projects MSA to Q, K, V
- Adds pair representation as attention bias
- Core innovation of AlphaFold 2

**`MSAColumnAttention`**: MSA column attention
- Transposes to attend across sequences
- Independent of pair representation

**`TriangleMultiplication`**: Triangle multiplicative update
- Gated projections for left and right edges
- Einstein summation for triangle computation
- Separate classes for outgoing/incoming

**`TriangleAttention`**: Triangle self-attention
- Standard multi-head attention over edges
- Two variants: starting and ending nodes

**`OuterProductMean`**: Outer product mean computation
- Projects MSA to lower dimension
- Computes outer product between positions
- Averages across MSA depth

**`EvoformerBlock`**: Complete Evoformer block
- Orchestrates all MSA and pair operations
- Includes layer norms and residual connections

**`TinyOpenFold`**: Main model class
- Input embeddings
- Stack of Evoformer blocks
- Structure module for predictions

### Data Flow

```
Input:
  ├─ MSA tokens (batch, n_seqs, seq_len)
  └─ Pair features (batch, seq_len, seq_len, pair_input_dim)

Embeddings:
  ├─ MSA: (batch, n_seqs, seq_len, msa_dim)
  └─ Pair: (batch, seq_len, seq_len, pair_dim)

Evoformer Blocks (repeated N times):
  ├─ MSA updates:
  │   ├─ Row attention (with pair bias)
  │   ├─ Column attention
  │   └─ Transition
  └─ Pair updates:
      ├─ Outer product mean
      ├─ Triangle multiplication (out/in)
      ├─ Triangle attention (start/end)
      └─ Transition

Structure Module:
  └─ Pair → Distances: (batch, seq_len, seq_len, 1)

Output:
  └─ Predicted distance matrix
```

## Debugging Tips

### Model Not Training (Loss Not Decreasing)

```bash
# Check with smaller problem
python tiny_openfold_v1.py \
    --seq-len 16 \
    --num-seqs 4 \
    --batch-size 2 \
    --num-steps 100

# Increase learning rate
python tiny_openfold_v1.py --learning-rate 1e-3
```

### Numerical Instabilities

```bash
# Use mixed precision for better numerical stability
python tiny_openfold_v1.py --use-amp
```

### Slow Performance

```bash
# Profile to find bottlenecks
python tiny_openfold_v1.py \
    --enable-pytorch-profiler \
    --profile-dir ./debug_profile \
    --num-steps 20

# Reduce problem size
python tiny_openfold_v1.py --seq-len 32 --num-seqs 8
```

## Understanding the Code

### Key Code Sections to Study

1. **MSA Row Attention** (lines ~250-310)
   - See how pair bias is added to attention scores
   - Note the broadcasting across MSA sequences

2. **Triangle Multiplication** (lines ~480-530)
   - Examine the Einstein summation for triangle updates
   - Understand gating mechanism

3. **Evoformer Block** (lines ~620-680)
   - See how MSA and pair updates are orchestrated
   - Note the residual connections

4. **Training Loop** (lines ~900-1050)
   - Profiling integration points
   - Timing and metrics collection

### Profiler Integration Points

The code includes `record_function()` calls for profiling:

```python
with record_function("evoformer_block"):
    with record_function("msa_row_attention"):
        # ... attention code
```

These show up in PyTorch Profiler and help identify bottlenecks.

## Next Steps

After running the baseline:

1. **Analyze Profiling Results**
   - Open TensorBoard to view timeline
   - Identify hotspot operations
   - Check memory usage patterns

2. **Experiment with Configurations**
   - Try different sequence lengths
   - Vary MSA depth
   - Test different numbers of blocks

3. **Consider Optimizations**
   - Implement flash attention for MSA operations
   - Fuse triangle operations
   - Try mixed precision training

## Resources

### AlphaFold 2 Paper
- Main: https://www.nature.com/articles/s41586-021-03819-2
- Supplement: Detailed architecture (Section 1.6 for Evoformer)

### OpenFold (Production Implementation)
- GitHub: https://github.com/aqlaboratory/openfold
- Documentation: https://openfold.readthedocs.io/

### Parent Directory
- See `../ARCHITECTURE.md` for detailed parameter calculations
- See `../README.md` for project overview

---

**Questions or Issues?**

Check the parent README or examine the code comments for detailed explanations of each component.

