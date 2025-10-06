# Version 1: PyTorch Baseline - Profiling Foundation

## Overview

Version 1 establishes the profiling foundation for the workshop using a standard PyTorch implementation of Tiny LLaMA. This version focuses on comprehensive performance characterization using PyTorch native profiling and DeepSpeed FLOPS profiler, providing the baseline measurements for all subsequent optimizations.

## Learning Objectives

After completing this version, you will be able to:
- Configure deterministic execution for reproducible profiling
- Use PyTorch Profiler for detailed operator-level analysis
- Integrate DeepSpeed FLOPS profiler for computational efficiency metrics
- Interpret profiling results and identify performance bottlenecks
- Establish baseline performance metrics for optimization comparison

## Architecture Overview

This implementation uses the standard transformer architecture with:
- **Multi-Head Attention**: Standard scaled dot-product attention
- **Feed-Forward Network**: SwiGLU activation with separate gate/up projections
- **Layer Normalization**: RMSNorm for improved training stability
- **Position Embeddings**: Rotary Position Embeddings (RoPE)

### Model Configuration

```python
# Default Tiny LLaMA Configuration
vocab_size = 1000           # Small vocabulary for workshop
hidden_size = 256          # Model dimension
num_layers = 4             # Transformer layers
num_attention_heads = 8    # Attention heads
intermediate_size = 512    # FFN dimension
max_sequence_length = 128  # Context window
```

## Implementation Details

### Mathematical Implementation

This section provides detailed implementation specifics for the baseline PyTorch model. For complete mathematical foundations, see [TINY_LLAMA_ARCHITECTURE.md](../TINY_LLAMA_ARCHITECTURE.md).

#### Standard PyTorch Attention Implementation

The baseline attention mechanism follows standard PyTorch patterns:

```python
def attention_forward(self, hidden_states, attention_mask=None):
    batch_size, seq_len, _ = hidden_states.size()

    # Linear projections (separate operations - optimization target!)
    query = self.q_proj(hidden_states)  # [B, S, D] -> [B, S, D]
    key = self.k_proj(hidden_states)    # [B, S, D] -> [B, S, D]
    value = self.v_proj(hidden_states)  # [B, S, D] -> [B, S, D]

    # Reshape for multi-head attention
    query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Apply rotary position embeddings
    query, key = self.rotary_emb(query, key, seq_len)

    # Scaled dot-product attention - O(S^2) memory complexity
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax over last dimension
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value)

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output
```

**Performance Characteristics:**
- **3 separate linear projections**: Creates kernel launch overhead
- **Attention matrix materialization**: $S \times S \times H$ memory usage
- **Multiple tensor reshapes**: Memory layout inefficiencies
- **Sequential operations**: Limited parallelization opportunities

#### SwiGLU Feed-Forward Implementation

```python
def swiglu_forward(self, hidden_states):
    # Separate gate and up projections (optimization target!)
    gate = self.gate_proj(hidden_states)  # [B, S, D] -> [B, S, D_ff]
    up = self.up_proj(hidden_states)      # [B, S, D] -> [B, S, D_ff]

    # SiLU activation (Swish)
    gate_activated = F.silu(gate)         # Element-wise operation

    # Element-wise multiplication
    intermediate = gate_activated * up     # [B, S, D_ff]

    # Down projection
    output = self.down_proj(intermediate)  # [B, S, D_ff] -> [B, S, D]

    return output
```

**Optimization Opportunities:**
- **Separate gate/up projections**: Can be fused into single GEMM
- **Intermediate tensor storage**: Memory overhead for gate_activated and up
- **Sequential activation**: SiLU can be fused with multiplication

#### RMSNorm Implementation

```python
def rms_norm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight * hidden_states).to(input_dtype)
```

**Implementation Details:**
- **Variance computation**: Single reduction operation
- **Epsilon for numerical stability**: Prevents division by zero
- **Mixed precision handling**: Maintains numerical precision

### Operator-Level Performance Analysis

#### FLOP Breakdown by Operation Type

```python
# Per transformer layer FLOP count (batch_size=1, seq_len=128)
FLOPS_BREAKDOWN = {
    'q_proj': seq_len * hidden_dim * hidden_dim,           # 128 * 256 * 256 = 8.4M
    'k_proj': seq_len * hidden_dim * hidden_dim,           # 128 * 256 * 256 = 8.4M
    'v_proj': seq_len * hidden_dim * hidden_dim,           # 128 * 256 * 256 = 8.4M
    'attn_scores': seq_len * seq_len * hidden_dim,         # 128 * 128 * 256 = 4.2M
    'attn_output': seq_len * seq_len * hidden_dim,         # 128 * 128 * 256 = 4.2M
    'o_proj': seq_len * hidden_dim * hidden_dim,           # 128 * 256 * 256 = 8.4M
    'gate_proj': seq_len * hidden_dim * intermediate_dim,  # 128 * 256 * 512 = 16.8M
    'up_proj': seq_len * hidden_dim * intermediate_dim,    # 128 * 256 * 512 = 16.8M
    'down_proj': seq_len * intermediate_dim * hidden_dim,  # 128 * 512 * 256 = 16.8M
    'rms_norm': 2 * seq_len * hidden_dim,                  # 2 * 128 * 256 = 65K
}

# Total per layer: ~92.1M FLOPs
# Total model (4 layers): ~368M FLOPs per forward pass
```

#### Memory Access Patterns

```python
# Memory bandwidth requirements per operation
MEMORY_BREAKDOWN = {
    'attention_qkv': {
        'parameters': 3 * hidden_dim * hidden_dim * 4,     # 3 * 256^2 * 4B = 786KB
        'activations': seq_len * hidden_dim * 4,           # 128 * 256 * 4B = 131KB
        'attention_matrix': seq_len * seq_len * num_heads * 4,  # 128^2 * 8 * 4B = 524KB
        'bandwidth_requirement': 'memory-bound'            # Limited by memory access
    },
    'feed_forward': {
        'parameters': 3 * hidden_dim * intermediate_dim * 4,    # 3 * 256 * 512 * 4B = 1.57MB
        'activations': seq_len * intermediate_dim * 4,          # 128 * 512 * 4B = 262KB
        'bandwidth_requirement': 'compute-bound'               # Good arithmetic intensity
    }
}
```

#### Kernel Launch Analysis

The baseline implementation generates numerous kernel launches per forward pass:

```python
# Typical kernel count per transformer layer
KERNEL_LAUNCHES = {
    'attention_block': {
        'q_projection': 1,        # Linear layer
        'k_projection': 1,        # Linear layer
        'v_projection': 1,        # Linear layer
        'rope_application': 2,    # For query and key
        'attention_computation': 3,  # QK^T, softmax, attention*V
        'output_projection': 1,   # Linear layer
        'residual_add': 1,        # Element-wise addition
        'subtotal': 10
    },
    'ffn_block': {
        'rms_norm': 1,           # Normalization
        'gate_projection': 1,     # Linear layer
        'up_projection': 1,       # Linear layer
        'silu_activation': 1,     # Element-wise SiLU
        'element_multiply': 1,    # gate * up
        'down_projection': 1,     # Linear layer
        'residual_add': 1,        # Element-wise addition
        'subtotal': 7
    },
    'layer_total': 17,           # Per transformer layer
    'model_total': 68            # 4 layers * 17 kernels/layer
}
```

**Optimization Implications:**
- **High kernel launch overhead**: 68+ kernels create GPU scheduling overhead
- **Memory bandwidth underutilization**: Many small operations
- **Fusion opportunities**: Adjacent operations can be combined

### Profiling Data Interpretation

#### PyTorch Profiler Output Analysis

When analyzing PyTorch profiler results, focus on these key metrics:

```python
# Key profiler metrics to examine
PROFILER_METRICS = {
    'operator_timing': {
        'aten::linear': 'Matrix multiplication operations',
        'aten::softmax': 'Attention softmax computation',
        'aten::add_': 'Residual connections',
        'aten::mul': 'Element-wise operations',
        'aten::rsqrt': 'RMSNorm operations'
    },
    'memory_analysis': {
        'peak_memory': 'Maximum GPU memory allocation',
        'memory_timeline': 'Memory usage over time',
        'fragmentation': 'Memory layout efficiency'
    },
    'gpu_utilization': {
        'kernel_efficiency': 'Individual kernel performance',
        'sm_efficiency': 'Streaming multiprocessor usage',
        'memory_bandwidth': 'Memory subsystem utilization'
    }
}
```

#### Expected Bottleneck Patterns

Based on the implementation analysis, expect these bottlenecks:

```python
EXPECTED_BOTTLENECKS = {
    'attention_computation': {
        'percentage_of_time': '35-45%',
        'primary_issue': 'O(S^{2}) memory complexity',
        'kernel_count': '10 per layer',
        'optimization_target': 'Flash Attention + QKV fusion'
    },
    'feed_forward_network': {
        'percentage_of_time': '30-40%',
        'primary_issue': 'Separate gate/up projections',
        'kernel_count': '7 per layer',
        'optimization_target': 'SwiGLU fusion'
    },
    'layer_normalization': {
        'percentage_of_time': '8-12%',
        'primary_issue': 'Memory-bound operation',
        'kernel_count': '2 per layer',
        'optimization_target': 'Kernel fusion with adjacent ops'
    },
    'residual_connections': {
        'percentage_of_time': '5-8%',
        'primary_issue': 'Memory bandwidth limitation',
        'kernel_count': '2 per layer',
        'optimization_target': 'Fusion with preceding operations'
    }
}
```

### Code Walkthrough: Critical Performance Paths

#### Attention Hot Path Analysis

```python
# Performance-critical code path in attention forward pass
@profile_function("attention_forward")  # PyTorch profiler annotation
def forward(self, hidden_states, attention_mask=None, position_ids=None):
    bsz, q_len, _ = hidden_states.size()

    # BOTTLENECK 1: Separate linear projections (3 kernel launches)
    with nvtx.range("qkv_projections"):
        query_states = self.q_proj(hidden_states)    # Kernel launch 1
        key_states = self.k_proj(hidden_states)      # Kernel launch 2
        value_states = self.v_proj(hidden_states)    # Kernel launch 3

    # Reshape for attention heads
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    # BOTTLENECK 2: Attention computation (O(S^2) memory)
    with nvtx.range("attention_computation"):
        # Attention scores: [bsz, num_heads, q_len, kv_seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # BOTTLENECK 3: Softmax (memory-bound)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # BOTTLENECK 4: Attention application
        attn_output = torch.matmul(attn_weights, value_states)

    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)  # Kernel launch 4

    return attn_output, attn_weights
```

**Profiling Annotations:**
- `@profile_function`: Enables detailed timing analysis
- `nvtx.range()`: Creates named regions in profiler traces
- Performance counters will show exact kernel timing

## Workshop Exercises

### Exercise 1: Baseline Performance Analysis

**Objective**: Establish baseline performance metrics and identify computational bottlenecks.

#### Step 1: Run Basic Training
```bash
# Basic training without profiling
python tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 10

# Expected output: Training loss progression and timing info
```

#### Step 2: Enable PyTorch Profiler
```bash
# Run with PyTorch profiler enabled
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 10 \
    --enable-pytorch-profiler \
    --profile-dir ./pytorch_profiles

# This generates detailed profiling traces in pytorch_profiles/
```

#### Step 3: Analyze Profiling Results
```bash
# Launch TensorBoard to visualize profiles
tensorboard --logdir pytorch_profiles --port 6006

# Or generate text report
python run_pytorch_profiler.py --analyze-existing pytorch_profiles/profile_*.json
```

**Expected Analysis Results:**
- Attention operations consuming ~40% of compute time
- Matrix multiplications (GEMM) as primary compute kernels
- Memory transfer overhead between operations
- GPU utilization patterns

#### Step 4: DeepSpeed FLOPS Analysis
```bash
# Run with DeepSpeed FLOPS profiler
python run_deepspeed_flops.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 10

# Analyze computational intensity
python run_deepspeed_flops.py --analyze-results flops_profile.json
```

**Expected FLOPS Analysis:**
- Total FLOPS per forward/backward pass
- FLOPS breakdown by operation type
- Model FLOPS Utilization (MFU) calculation
- Memory bandwidth requirements

### Exercise 2: Memory Analysis and Optimization

**Objective**: Understand memory usage patterns and bandwidth requirements.

#### Step 1: Memory Profiling
```bash
# Run with memory profiling enabled
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --enable-pytorch-profiler \
    --profile-memory \
    --profile-dir ./memory_analysis

# Generate memory timeline visualization
python -c "
import torch
from torch.profiler import profile, record_function, ProfilerActivity
# Memory analysis code will be embedded in tiny_llama_v1.py
"
```

#### Step 2: Batch Size Scaling
```bash
# Test different batch sizes
for bs in 4 8 16 32; do
    echo \"Testing batch size: \$bs\"
    python tiny_llama_v1.py \
        --batch-size \$bs \
        --seq-len 128 \
        --num-steps 5 \
        --enable-pytorch-profiler \
        --profile-dir ./scaling_bs\$bs
done

# Analyze scaling behavior
python analyze_batch_scaling.py --profile-dirs scaling_bs*
```

**Expected Memory Analysis:**
- Memory usage scaling with batch size
- Peak memory allocation points
- Memory fragmentation patterns
- Opportunities for memory optimization

### Exercise 3: Bottleneck Identification

**Objective**: Identify computational and memory bottlenecks for optimization targets.

#### Step 1: Operator-Level Analysis
```bash
# Detailed operator timing
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --enable-pytorch-profiler \
    --profile-operators \
    --sort-by cuda_time_total

# Generate bottleneck report
python analyze_bottlenecks.py \
    --profile-data pytorch_profiles/ \
    --output-report bottlenecks_v1.md
```

#### Step 2: Attention Pattern Analysis
```bash
# Focus on attention computation
python tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --profile-attention-only \
    --enable-pytorch-profiler
```

#### Step 3: Matrix Multiplication Analysis
```bash
# GEMM operation profiling
python analyze_gemm_operations.py \
    --model-config tiny_llama_v1_config.yaml \
    --batch-sizes \"4,8,16,32\" \
    --sequence-lengths \"64,128,256\"
```

**Expected Bottleneck Analysis:**
- Attention QKV projection overhead
- Softmax computation inefficiency
- Multiple small GEMM operations
- Memory-bound operations identification

## Profiling Tools Integration

### PyTorch Profiler Configuration

The implementation includes comprehensive PyTorch profiler integration:

```python
# In tiny_llama_v1.py
from torch.profiler import profile, record_function, ProfilerActivity

# Profiler configuration
profiler_config = {
    'activities': [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    'record_shapes': True,
    'profile_memory': True,
    'with_stack': True,
    'with_flops': True,
    'experimental_config': torch._C._profiler._ExperimentalConfig(verbose=True)
}
```

### DeepSpeed FLOPS Profiler Integration

```python
# FLOPS profiler setup
from deepspeed.profiling.flops_profiler import FlopsProfiler

profiler = FlopsProfiler(model)
profiler.start_profile()
# Training step
profiler.stop_profile()
profiler.print_model_profile(profile_step=1)
```

## Key Performance Metrics

### Baseline Performance Expectations

On a typical AMD MI200 series GPU:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| **Training Speed** | 50-100 samples/sec | Batch size dependent |
| **GPU Utilization** | 60-75% | Standard PyTorch efficiency |
| **Memory Usage** | 2-4 GB | Model + batch data |
| **FLOPS Utilization** | 30-45% | Baseline MFU |
| **Memory Bandwidth** | 40-60% | Memory-bound operations |

### Profiling Output Files

After running exercises, expect these output files:

```
version1_pytorch_baseline/
├── pytorch_profiles/
│   ├── profile_*.json          # PyTorch profiler traces
│   ├── trace_*.json            # Chrome trace format
│   └── memory_timeline.html    # Memory usage visualization
├── flops_analysis/
│   ├── flops_profile.json      # FLOPS breakdown
│   ├── model_profile.txt       # Detailed model analysis
│   └── mfu_analysis.csv        # Model FLOPS Utilization
└── bottleneck_analysis/
    ├── bottlenecks_v1.md       # Comprehensive bottleneck report
    ├── operator_timing.csv     # Per-operator performance
    └── optimization_targets.json # Prioritized optimization opportunities
```

## Expected Analysis Results

### Performance Characteristics

1. **Compute Distribution**:
   - Attention operations: ~40% of total time
   - Feed-forward network: ~35% of total time
   - Layer normalization: ~10% of total time
   - Other operations: ~15% of total time

2. **Memory Patterns**:
   - Peak memory usage during attention computation
   - Multiple intermediate tensor allocations
   - Memory fragmentation from varying tensor sizes

3. **Optimization Opportunities**:
   - Kernel fusion potential in attention
   - Memory layout optimization
   - Reduced intermediate tensor creation

### Bottleneck Identification

Primary bottlenecks to address in subsequent versions:

1. **Separate QKV projections** → Fusion opportunity
2. **Standard attention computation** → Flash Attention
3. **Individual FFN gates** → SwiGLU fusion
4. **Multiple kernel launches** → Custom kernels

## Troubleshooting

### Common Issues

#### CUDA/ROCm Memory Errors
```bash
# Reduce batch size if memory errors occur
python tiny_llama_v1.py --batch-size 4 --seq-len 64
```

#### Profiler Permission Issues
```bash
# Ensure proper permissions for profiling
export ROCPROF_COMPUTE_DISABLE_AQL_DEBUG=1
```

#### Missing Profiling Output
```bash
# Check profiling directory permissions
mkdir -p pytorch_profiles
chmod 755 pytorch_profiles
```

### Performance Validation

To validate your setup is working correctly:

```bash
# Quick validation run
python tiny_llama_v1.py \
    --batch-size 4 \
    --seq-len 64 \
    --num-steps 3 \
    --enable-pytorch-profiler \
    --validate-setup

# Expected: Successful completion with profiling files generated
```

## Next Steps

After completing all exercises in Version 1:

1. **Review baseline metrics** - Understand current performance characteristics
2. **Identify optimization targets** - Use bottleneck analysis to prioritize improvements
3. **Prepare for Version 2** - Kernel fusion will address primary bottlenecks
4. **Document findings** - Record baseline measurements for comparison

**Ready for optimization? Proceed to [Version 2: PyTorch Fused](../version2_pytorch_fused/README.md)**

---

## Performance Summary Template

Use this template to document your Version 1 results:

```
# Version 1 Baseline Results

## Configuration
- Batch Size: ___
- Sequence Length: ___
- GPU: ___
- ROCm Version: ___

## Performance Metrics
- Training Speed: ___ samples/sec
- GPU Utilization: ___%
- Memory Usage: ___ GB
- FLOPS Utilization: ___%

## Top Bottlenecks
1. _________________ (__% of time)
2. _________________ (__% of time)
3. _________________ (__% of time)

## Optimization Targets for Version 2
- [ ] QKV fusion
- [ ] Flash Attention
- [ ] SwiGLU fusion
- [ ] Other: ___________
```