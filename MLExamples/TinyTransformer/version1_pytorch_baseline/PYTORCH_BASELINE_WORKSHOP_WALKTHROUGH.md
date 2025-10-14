# Tiny LLaMA PyTorch Baseline - Profiling Workshop
## Complete Hands-On Walkthrough Guide

---

## Important Note

**The performance numbers and metrics shown throughout this workshop are representative examples and were collected on specific hardware configurations.** Your actual results will differ based on:

- GPU model (e.g., MI250X, MI300X, MI325X)
- ROCm version
- PyTorch version
- System configuration (CPU, memory, drivers)
- Current GPU utilization and temperature

**Focus on the relative improvements and optimization techniques** demonstrated in each exercise rather than matching the exact numbers shown. The methodologies and analysis approaches are applicable across different hardware platforms.

---

## Notation and Variables

Throughout this workshop, we use the following notation for tensor dimensions and model parameters:

**Tensor Dimensions:**
- **B** = Batch size (number of samples processed together)
- **S** = Sequence length (number of tokens in each sequence)
- **D** = Hidden dimension / Model dimension (size of hidden representations)
- **H** = Number of attention heads
- **head_dim** = Dimension per attention head (typically D / H)

**Model Parameters:**
- **D_ff** = Feed-forward network intermediate dimension
- **V** = Vocabulary size (number of unique tokens)
- **L** = Number of transformer layers

**Performance Metrics:**
- **FLOPS** = Floating Point Operations Per Second
- **MFU** = Model FLOPS Utilization (% of theoretical peak achieved)
- **TFLOPS** = Tera-FLOPS (10^12 floating point operations per second)
- **GFLOPS** = Giga-FLOPS (10^9 floating point operations per second)

**Complexity Notation:**
- **O(S)** = Linear complexity with sequence length
- **O(S^2)** = Quadratic complexity with sequence length
- **O(B × S × D)** = Complexity grows with batch, sequence, and dimension

**Example Tensor Shapes:**
```
Input tensor:           [B, S, D]      e.g., [8, 128, 256]
Attention weights:      [B, H, S, S]   e.g., [8, 8, 128, 128]
Query/Key/Value:        [B, H, S, head_dim] e.g., [8, 8, 128, 32]
FFN intermediate:       [B, S, D_ff]   e.g., [8, 128, 512]
```

---

## Table of Contents

1. [Introduction & Setup](#1-introduction--setup)
2. [Understanding Tiny LLaMA Architecture](#2-understanding-tiny-llama-architecture)
3. [Understanding the Baseline Implementation](#3-understanding-the-baseline-implementation)
4. [Exercise 1: Baseline Performance Analysis](#4-exercise-1-baseline-performance-analysis)
5. [Exercise 2: Memory Analysis & Optimization](#5-exercise-2-memory-analysis--optimization)
6. [Exercise 3: Performance Study Across Problem Sizes](#6-exercise-3-performance-study-across-problem-sizes)

---

## 1. Introduction & Setup

### 1.1 What is LLM Training?

**Large Language Model (LLM) Training** involves teaching neural networks to understand and generate human language through iterative optimization of model parameters.

**Key Differences: Training vs Inference**

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Purpose** | Learn patterns from data | Make predictions |
| **Direction** | Forward + Backward pass | Forward pass only |
| **Gradients** | Required and computed | Not required |
| **Batch Size** | Typically larger (8-64) | Often smaller (1-32) |
| **Performance Goal** | Samples/sec + FLOPS efficiency | Latency + throughput |
| **Memory Usage** | Very high (activations + gradients) | Lower (no gradient storage) |
| **Optimization Focus** | Throughput, MFU, memory efficiency | Latency, batch throughput |

**Why Profile LLM Training?**

- Understand computational bottlenecks
- Optimize hardware utilization (Model FLOPS Utilization - MFU)
- Reduce training costs
- Identify memory inefficiencies
- Guide optimization decisions
- Establish baseline for improvements

### 1.2 Workshop Goals

By the end of this workshop, you will be able to:

- Configure and run deterministic PyTorch LLM training
- Use PyTorch Profiler for detailed operator-level analysis
- Integrate DeepSpeed FLOPS profiler for computational efficiency metrics
- Interpret profiling results and identify performance bottlenecks
- Understand memory usage patterns in transformer training
- Analyze attention mechanisms and FFN performance
- Calculate Model FLOPS Utilization (MFU)
- Establish baseline performance metrics for optimization comparison

### 1.3 Understanding Key Metrics

Before diving into exercises, let's understand the metrics we'll be measuring:

#### Training Speed (samples/sec)
- **What:** Number of training samples processed per second
- **Higher is better**
- **Typical range:** 50-200 samples/sec for small models on single GPU
- **Formula:** `(batch_size × num_steps) / total_time`

#### FLOPS (Floating Point Operations Per Second)
- **What:** Computational throughput
- **Higher is better**
- **Units:** TFLOPS (TeraFLOPS, 10^12 operations/second)
- **Theoretical Peak:** Hardware maximum (e.g., MI250X: ~95 TFLOPS FP32, ~190 TFLOPS FP16)

#### Model FLOPS Utilization (MFU)
- **What:** Percentage of theoretical peak FLOPS achieved
- **Formula:** `(Achieved FLOPS / Theoretical Peak FLOPS) × 100%`
- **Typical ranges:**
  - 20-30%: Baseline PyTorch (memory-bound)
  - 40-50%: Well-optimized (compute-bound)
  - 60%+: Highly optimized (kernel fusion, Flash Attention)

#### Memory Usage (GB)
- **What:** GPU memory consumed
- **Components:** Model weights + optimizer states + activations + gradients
- **Lower is better** (allows larger batches)

#### GPU Utilization (%)
- **What:** Percentage of GPU compute units in use
- **Higher is better** (approaching 100%)
- **Low utilization indicates:** Memory bottlenecks, CPU bottlenecks, or small workloads

### 1.4 Environment Verification

Let's verify your system is ready for the workshop.

#### Step 1: Check ROCm Installation

```bash
# Check if ROCm is installed
rocminfo | grep "Name:"
```

**Expected Output:**
```
  Name:                    gfx90a
  Name:                    AMD Instinct MI250X
```

**If you see an error:**
```bash
# Check if ROCm is installed
which rocminfo

# If not found, ROCm is not installed
# Contact your system administrator
```

#### Step 2: Check GPU Visibility

```bash
# Check GPU status
rocm-smi
```

**Expected Output:**
```
GPU[0]    : GPU ID: 0
GPU[0]    : GPU Name: AMD Instinct MI250X
GPU[0]    : Temperature: 35.0°C
GPU[0]    : GPU Memory Usage: 512 MB / 65536 MB
GPU[0]    : GPU Utilization: 0%
```

**Common Issues:**

**Error: "Unable to detect any GPUs"**
```bash
# Check permissions
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Logout and login again
# Then retry: rocm-smi
```

#### Step 3: Check PyTorch + ROCm

```bash
# Test PyTorch with ROCm
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('ERROR: No GPU detected!')
"
```

**Expected Output:**
```
PyTorch Version: 2.7.1+rocm6.4.4
CUDA Available: True
GPU Name: AMD Instinct MI250X
GPU Memory: 65.5 GB
```

**Common Issues:**

**Error: "ModuleNotFoundError: No module named 'torch'"**
```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Error: "CUDA Available: False"**
```bash
# Check if ROCm-enabled PyTorch is installed
python3 -c "import torch; print(torch.__version__)"

# Should show something like: 2.7.1+rocm6.4.4
# If it shows 2.7.1+cpu, you have CPU-only PyTorch

# Reinstall with ROCm support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

#### Step 4: Check DeepSpeed (Optional but Recommended)

```bash
# Check if DeepSpeed is installed
python3 -c "import deepspeed; print(f'DeepSpeed Version: {deepspeed.__version__}')"
```

**Expected Output:**
```
DeepSpeed Version: 0.12.6
```

**If not installed:**
```bash
# Install DeepSpeed
pip install deepspeed
```

#### Step 5: Navigate to Workshop Directory

```bash
# Navigate to version1_pytorch_baseline directory
cd ~/castille-ai-workshop-training/version1_pytorch_baseline/

# List files
ls -la
```

**Expected Output:**
```
-rw-rw-r-- tiny_llama_v1.py
-rw-rw-r-- run_pytorch_profiler.py
-rw-rw-r-- run_deepspeed_flops.py
-rw-rw-r-- README.md
-rwxrwxr-x run_baseline.sh
-rwxrwxr-x run_pytorch_profiler.sh
-rwxrwxr-x run_deepspeed_flops.sh
drwxrwxr-x exercises/
```

#### Step 6: Quick Test Run

Let's verify everything works with a very small test:

```bash
# Run a tiny test (should complete in ~1-2 minutes)
python3 tiny_llama_v1.py --batch-size 4 --seq-len 64 --num-steps 5
```

**Expected Output:**
```
==========================================
Tiny LLaMA V1 - PyTorch Baseline
==========================================
Configuration:
  Batch Size: 4
  Sequence Length: 64
  Number of Steps: 5
  Hidden Dim: 256
  Num Layers: 4
  Num Heads: 8

Initializing model...
Model parameters: 2.3M

Starting training...
Step 1/5: Loss = 6.9088, Time = 0.235 seconds
Step 2/5: Loss = 6.9076, Time = 0.045 seconds
Step 3/5: Loss = 6.9065, Time = 0.044 seconds
Step 4/5: Loss = 6.9054, Time = 0.043 seconds
Step 5/5: Loss = 6.9042, Time = 0.043 seconds

==========================================
Performance Summary:
==========================================
Average time per step: 0.044 seconds
Training speed: 90.9 samples/sec
Peak memory usage: 1234 MB
==========================================
```

**If you see this output, your environment is ready!**


---

## 2. Understanding Tiny LLaMA Architecture

### 2.1 Model Overview

Tiny LLaMA is a scaled-down version of the LLaMA architecture, designed for educational purposes and profiling workshops. It uses the standard transformer decoder architecture with modern enhancements.

**Model Configuration (Default):**

```python
vocab_size = 1000           # Small vocabulary for workshop
hidden_dim = 256            # Model dimension (D)
n_layers = 4                # Number of transformer layers
n_heads = 8                 # Number of attention heads
n_kv_heads = 4              # Number of key-value heads (GQA)
intermediate_dim = 512      # FFN intermediate dimension
max_seq_len = 128           # Maximum sequence length
```

**Model Size:**
- Parameters: ~2.9 million
- Memory footprint: ~11 MB (FP32)
- Training memory (batch=8, seq=128): ~200-500 MB (includes activations, gradients, optimizer states)

**Detailed Parameter Calculation:**

Understanding how we arrive at ~2.9M parameters:

1. **Token Embeddings**:
   - Shape: [vocab_size, hidden_dim] = [1000, 256]
   - Parameters: 1000 × 256 = 256,000

2. **Per Transformer Layer** (4 layers total):

   a. **RMSNorm (×2 per layer)**:
      - Pre-attention norm: hidden_dim = 256 parameters
      - Pre-FFN norm: hidden_dim = 256 parameters
      - Total: 2 × 256 = 512 parameters per layer

   b. **Multi-Head Attention with GQA** (Grouped Query Attention):
      - **Q projection**: [hidden_dim, hidden_dim] = [256, 256] = 65,536 parameters
      - **K projection** (GQA): [hidden_dim, head_dim × n_kv_heads] = [256, 32 × 4] = [256, 128] = 32,768 parameters
        - Why smaller? GQA uses fewer key/value heads (4) than query heads (8)
        - head_dim = hidden_dim / n_heads = 256 / 8 = 32
      - **V projection** (GQA): [256, 128] = 32,768 parameters
      - **O projection** (output): [256, 256] = 65,536 parameters
      - **Total Attention**: 65,536 + 32,768 + 32,768 + 65,536 = 196,608 parameters per layer

   c. **SwiGLU Feed-Forward Network**:
      - **Gate projection**: [hidden_dim, intermediate_dim] = [256, 512] = 131,072 parameters
      - **Up projection**: [256, 512] = 131,072 parameters
      - **Down projection**: [intermediate_dim, hidden_dim] = [512, 256] = 131,072 parameters
      - **Total FFN**: 131,072 + 131,072 + 131,072 = 393,216 parameters per layer

   d. **Total per layer**: 512 + 196,608 + 393,216 = 590,336 parameters

   e. **All 4 layers**: 4 × 590,336 = 2,361,344 parameters

3. **Final Components**:
   - **Final RMSNorm**: 256 parameters
   - **Output projection** (LM head): [hidden_dim, vocab_size] = [256, 1000] = 256,000 parameters
   - **Total**: 256 + 256,000 = 256,256 parameters

4. **Grand Total**:
   - Embeddings: 256,000
   - All layers: 2,361,344
   - Final components: 256,256
   - **Total**: 256,000 + 2,361,344 + 256,256 = **2,873,600 parameters ≈ 2.9M**

**Memory Footprint Calculation:**
- FP32: 4 bytes per parameter
- Total memory: 2,873,600 × 4 bytes = 11,494,400 bytes ≈ **11.0 MB**

**Training Memory Breakdown** (batch_size=8, seq_len=128):

Per-layer memory requirements:
- **Input activations**: [B, S, D] = [8, 128, 256] = 262,144 elements → 1.05 MB
- **Q, K, V tensors**: 3 × [8, 128, 256] → 3.15 MB
- **Attention scores**: [B, H, S, S] = [8, 8, 128, 128] = 1,048,576 elements → 4.19 MB
- **FFN intermediates**: 2 × [B, S, D_ff] = 2 × [8, 128, 512] → 4.19 MB
- **Per-layer subtotal**: ~15.7 MB × 4 layers = **~63 MB**

Training overhead:
- **Gradients** (same size as activations): ~63 MB
- **Parameter gradients**: 2.9M × 4 bytes = ~11 MB
- **Optimizer states** (Adam: momentum + variance): 2.9M × 2 × 4 bytes = ~22 MB

**Total training memory**: 63 + 63 + 11 + 22 = **~160 MB**

Note: Actual PyTorch memory usage will be 200-500 MB due to:
- Framework overhead
- Memory fragmentation
- Temporary buffers
- CUDA kernels and workspace

### 2.2 Transformer Layer Architecture

Each transformer layer consists of:

1. **RMSNorm** (Root Mean Square Normalization)
2. **Multi-Head Attention** with RoPE
3. **Residual Connection**
4. **RMSNorm**
5. **Feed-Forward Network** (SwiGLU)
6. **Residual Connection**

**Visual Structure:**

```
Input (B, S, D)
    ↓
┌───────────────────────────────────────┐
│  RMSNorm                              │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  Multi-Head Attention                 │
│  ┌─────────────────────────────────┐ │
│  │ Q, K, V Projections             │ │
│  │ RoPE (Rotary Position Encoding) │ │
│  │ Attention Computation           │ │
│  │ Output Projection               │ │
│  └─────────────────────────────────┘ │
└───────────────────────────────────────┘
    ↓
  Residual Add
    ↓
┌───────────────────────────────────────┐
│  RMSNorm                              │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  Feed-Forward Network (SwiGLU)        │
│  ┌─────────────────────────────────┐ │
│  │ Gate Projection                 │ │
│  │ Up Projection                   │ │
│  │ SiLU Activation                 │ │
│  │ Element-wise Multiply           │ │
│  │ Down Projection                 │ │
│  └─────────────────────────────────┘ │
└───────────────────────────────────────┘
    ↓
  Residual Add
    ↓
Output (B, S, D)
```

### 2.3 Multi-Head Attention Implementation

**Standard PyTorch Attention (Version 1 Baseline):**

The baseline uses separate linear projections for Query, Key, and Value:

```python
def attention_forward(self, hidden_states, attention_mask=None):
    batch_size, seq_len, _ = hidden_states.size()

    # STEP 1: Separate linear projections (3 kernel launches)
    query = self.q_proj(hidden_states)  # [B, S, D] -> [B, S, D]
    key = self.k_proj(hidden_states)    # [B, S, D] -> [B, S, D]
    value = self.v_proj(hidden_states)  # [B, S, D] -> [B, S, D]

    # STEP 2: Reshape for multi-head attention
    query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    # Result: [B, H, S, head_dim]

    # STEP 3: Apply rotary position embeddings
    query, key = self.rotary_emb(query, key, seq_len)

    # STEP 4: Compute attention scores
    # attn_weights: [B, H, S, S]
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # STEP 5: Softmax over last dimension
    attn_weights = F.softmax(attn_weights, dim=-1)

    # STEP 6: Apply attention to values
    attn_output = torch.matmul(attn_weights, value)
    # Result: [B, H, S, head_dim]

    # STEP 7: Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output
```

**Performance Characteristics:**

- **3 separate linear projections:** Creates kernel launch overhead
- **Attention matrix materialization:** O(S^2) memory usage per head
- **Memory-bound operations:** Multiple tensor reshapes
- **Sequential execution:** Limited parallelization

**FLOP Count (per layer):**

Understanding FLOP calculations for attention operations with example configuration (B=8, S=128, D=256, H=8, head_dim=32):

**Linear Projection FLOP Formula:**
For a matrix multiplication: `output = input @ weight`
- Input shape: [B, S, D_in]
- Weight shape: [D_in, D_out]
- FLOPs = 2 × B × S × D_in × D_out
  - Factor of 2: Each multiply-accumulate (MAC) operation counts as 2 FLOPs (1 multiply + 1 add)
  - We perform B × S output positions, each requiring D_in × D_out operations

**Attention FLOP Calculations:**

1. **Q, K, V Projections** (3 separate linear layers):
   - Each projection: [B, S, D] → [B, S, D]
   - FLOPs per projection: 2 × B × S × D × D
   - Calculation: 2 × 8 × 128 × 256 × 256 = 134,217,728 ≈ 134.2M FLOPs
   - Total for Q, K, V: 3 × 134.2M = 402.6M FLOPs

2. **Attention Scores** (Q @ K^T):
   - After reshaping: Q and K are [B, H, S, head_dim]
   - For each head: [S, head_dim] @ [head_dim, S] → [S, S]
   - FLOPs: 2 × B × H × S × S × head_dim
   - Calculation: 2 × 8 × 8 × 128 × 128 × 32 = 67,108,864 ≈ 67.1M FLOPs
   - Why: For each of B×H attention matrices, we compute S×S scores, each requiring head_dim multiply-accumulates

3. **Attention Application** (Softmax @ V):
   - Attention weights [B, H, S, S] @ Values [B, H, S, head_dim] → [B, H, S, head_dim]
   - FLOPs: 2 × B × H × S × S × head_dim
   - Calculation: 2 × 8 × 8 × 128 × 128 × 32 = 67.1M FLOPs
   - Same as attention scores computation

4. **Output Projection**:
   - [B, S, D] → [B, S, D]
   - FLOPs: 2 × B × S × D × D
   - Calculation: 2 × 8 × 128 × 256 × 256 = 134.2M FLOPs

**Summary:**
```
Q projection:           134.2M FLOPs
K projection:           134.2M FLOPs
V projection:           134.2M FLOPs
Attention scores:        67.1M FLOPs
Softmax:                 ~0.1M FLOPs (negligible, element-wise)
Attention application:   67.1M FLOPs
Output projection:      134.2M FLOPs
─────────────────────────────────
Total Attention:       ~671M FLOPs per layer
```

**Key Insights:**
- Linear projections (Q, K, V, O) dominate: 536.8M FLOPs (80% of attention)
- Attention computation (scores + application): 134.2M FLOPs (20% of attention)
- Quadratic term (S × S) appears in attention scores but with small head_dim coefficient
- For longer sequences, the S^2 term becomes more significant

### 2.4 SwiGLU Feed-Forward Network

**Implementation:**

```python
def swiglu_forward(self, hidden_states):
    # STEP 1: Separate gate and up projections (2 kernel launches)
    gate = self.gate_proj(hidden_states)  # [B, S, D] -> [B, S, D_ff]
    up = self.up_proj(hidden_states)      # [B, S, D] -> [B, S, D_ff]

    # STEP 2: SiLU activation (Swish)
    gate_activated = F.silu(gate)         # Element-wise operation

    # STEP 3: Element-wise multiplication
    intermediate = gate_activated * up     # [B, S, D_ff]

    # STEP 4: Down projection
    output = self.down_proj(intermediate)  # [B, S, D_ff] -> [B, S, D]

    return output
```

**Why SwiGLU?**
- Better than standard ReLU activation
- Gating mechanism improves model capacity
- Used in modern LLMs (LLaMA, PaLM)

**Performance Characteristics:**
- **Separate gate/up projections:** Can be fused into single GEMM
- **Intermediate tensor storage:** Memory overhead
- **Sequential activation:** SiLU can be fused with multiplication

**FLOP Count (per layer):**

Understanding FLOP calculations for feed-forward network with example configuration (B=8, S=128, D=256, D_ff=512):

**FFN FLOP Calculations:**

1. **Gate Projection**:
   - Transform: [B, S, D] → [B, S, D_ff]
   - Weight matrix: [D, D_ff] = [256, 512]
   - FLOPs: 2 × B × S × D × D_ff
   - Calculation: 2 × 8 × 128 × 256 × 512 = 268,435,456 ≈ 268.4M FLOPs
   - Explanation: For each of B×S positions, multiply a D-dimensional vector by a [D, D_ff] matrix

2. **Up Projection**:
   - Same dimensions as gate projection: [B, S, D] → [B, S, D_ff]
   - FLOPs: 2 × B × S × D × D_ff = 268.4M FLOPs
   - Calculation: 2 × 8 × 128 × 256 × 512 = 268.4M FLOPs

3. **SiLU Activation**:
   - Element-wise operation: silu(x) = x × sigmoid(x)
   - Applied to gate tensor: [B, S, D_ff]
   - FLOPs: ~3 × B × S × D_ff (sigmoid + multiply) ≈ 0.01M FLOPs
   - Negligible compared to matrix multiplications

4. **Element-wise Multiply**:
   - gate_activated × up: [B, S, D_ff] element-wise
   - FLOPs: B × S × D_ff = 8 × 128 × 512 ≈ 0.5M FLOPs
   - Negligible compared to linear projections

5. **Down Projection**:
   - Transform: [B, S, D_ff] → [B, S, D]
   - Weight matrix: [D_ff, D] = [512, 256]
   - FLOPs: 2 × B × S × D_ff × D
   - Calculation: 2 × 8 × 128 × 512 × 256 = 268,435,456 ≈ 268.4M FLOPs

**Summary:**
```
Gate projection:        268.4M FLOPs
Up projection:          268.4M FLOPs
Down projection:        268.4M FLOPs
SiLU activation:         ~0.01M FLOPs (negligible)
Element-wise multiply:   ~0.5M FLOPs (negligible)
─────────────────────────────────
Total FFN:             ~805.3M FLOPs per layer
```

**Key Insights:**
- Three linear projections dominate: 805.2M FLOPs (>99.9% of FFN)
- Element-wise operations (SiLU, multiply) are negligible: <1M FLOPs combined
- FFN is more compute-intensive than attention: 805M vs 671M FLOPs
- Gate and up projections can be fused to reduce memory bandwidth
- D_ff is typically 2-4× larger than D, making FFN compute-bound

### 2.5 RMSNorm (Root Mean Square Normalization)

**Implementation:**

```python
def rms_norm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype

    # Compute RMS
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

    # Apply learned scale
    return (self.weight * hidden_states).to(input_dtype)
```

**Why RMSNorm instead of LayerNorm?**
- Simpler: No mean subtraction
- Faster: Fewer operations
- Same effectiveness for LLMs
- Less memory bandwidth

**Performance Characteristics:**
- Memory-bound operation
- Reduction over hidden dimension
- Opportunity for fusion with adjacent operations

### 2.6 Complete Layer FLOP Breakdown

For a single transformer layer with batch_size=8, seq_len=128:

```
Component               | FLOPs        | Percentage
------------------------|--------------|------------
Attention QKV Proj      | 402.6M       | 27.3%
Attention Computation   | 134.2M       | 9.1%
Attention Output Proj   | 134.2M       | 9.1%
FFN Gate/Up Proj        | 536.8M       | 36.4%
FFN Down Proj           | 268.4M       | 18.2%
RMSNorm (x2)            | ~0.5M        | <0.1%
------------------------|--------------|------------
Total per Layer         | ~1,476M      | 100%
Total Model (4 layers)  | ~5.91B       | -
```

**Corrected Calculations:**
- Attention QKV: 3 × 134.2M = 402.6M FLOPs
- Attention scores + application: 67.1M + 67.1M = 134.2M FLOPs
- Attention output: 134.2M FLOPs
- FFN gate + up: 2 × 268.4M = 536.8M FLOPs
- FFN down: 268.4M FLOPs
- Total per layer: 402.6 + 134.2 + 134.2 + 536.8 + 268.4 + 0.5 = 1,476.7M ≈ 1.48B FLOPs
- Total model (4 layers): 4 × 1.48B = 5.92B FLOPs per forward pass

**Key Observations:**
- FFN dominates compute: ~54.6% of FLOPs (gate/up/down projections)
- Attention: ~45.5% of FLOPs
- RMSNorm negligible: <0.1% of FLOPs
- Linear projections (GEMM operations) account for >99% of all FLOPs

### 2.7 Memory Layout and Access Patterns

**Memory Requirements (batch_size=8, seq_len=128):**

```
Component              | Memory (MB) | Notes
-----------------------|-------------|---------------------------
Model Parameters       | 9.2         | Weights only (FP32)
Optimizer States       | 36.8        | Adam: 2× params (m, v)
Input Activations      | 1.0         | Per layer
Attention Activations  | 4.2         | Intermediate tensors
FFN Activations        | 2.1         | Intermediate tensors
Gradients              | 9.2         | Same as parameters
Attention Matrix       | 1.0         | [B, H, S, S] per layer
-----------------------|-------------|---------------------------
Total (approximate)    | 63.5 MB     | Can vary with framework
```

**Memory Bandwidth Patterns:**

- **Attention:** Memory-bound (many small operations, reshapes)
- **FFN:** Compute-bound (large GEMMs with high arithmetic intensity)
- **RMSNorm:** Memory-bound (reduction operations)

---

## 3. Understanding the Baseline Implementation

### 3.1 Code Structure Overview

The `tiny_llama_v1.py` file is organized into several key components:

```
tiny_llama_v1.py
├── Configuration Classes
│   ├── TinyLlamaConfig (model configuration)
│   └── ProfilerConfig (profiling options)
├── Model Components
│   ├── RMSNorm (normalization layer)
│   ├── RotaryEmbedding (position encoding)
│   ├── Attention (multi-head attention)
│   ├── MLP (SwiGLU feed-forward)
│   ├── TransformerBlock (complete layer)
│   └── TinyLlamaModel (full model)
├── Training Infrastructure
│   ├── Optimizer setup
│   ├── Loss computation
│   └── Training loop
└── Profiling Integration
    ├── PyTorch Profiler setup
    ├── DeepSpeed FLOPS profiler
    └── Performance reporting
```

### 3.2 Command-Line Arguments

Understanding the available options:

**Basic Training Arguments:**

```bash
--batch-size 8              # Number of samples per batch
--seq-len 128               # Sequence length
--num-steps 50              # Number of training steps
--learning-rate 1e-4        # Optimizer learning rate
--device cuda               # Device to use (cuda/cpu)
```

**Model Configuration:**

```bash
--hidden-dim 256            # Model hidden dimension
--n-layers 4                # Number of transformer layers
--n-heads 8                 # Number of attention heads
--intermediate-dim 512      # FFN intermediate size
```

**Profiling Options:**

```bash
--enable-pytorch-profiler   # Enable PyTorch profiler
--profile-dir ./profiles    # Directory for profile output
--profile-memory            # Include memory profiling
--profile-operators         # Detailed operator profiling
--profile-steps 5           # Number of steps to profile
```

**DeepSpeed FLOPS Profiling:**

```bash
--enable-deepspeed-flops    # Enable FLOPS profiler
--flops-profile-step 10     # Which step to profile
```

**Other Options:**

```bash
--seed 42                   # Random seed for reproducibility
--deterministic             # Enable deterministic operations
--output-dir ./output       # Directory for outputs
--log-interval 10           # Logging frequency
```

### 3.3 Profiling Integration Points

The code includes several profiling integration points:

**PyTorch Profiler Context:**

```python
# In training loop
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    # Training step
    outputs = model(inputs)
    loss = criterion(outputs)
    loss.backward()
    optimizer.step()

# Export results
prof.export_chrome_trace("trace.json")
```

**NVTX Annotations:**

```python
# Mark important regions
with nvtx.range("attention_forward"):
    attn_output = attention(hidden_states)

with nvtx.range("ffn_forward"):
    ffn_output = feed_forward(hidden_states)
```

**DeepSpeed FLOPS Profiler:**

```python
from deepspeed.profiling.flops_profiler import FlopsProfiler

profiler = FlopsProfiler(model)
profiler.start_profile()
# Forward pass
profiler.stop_profile()
profiler.print_model_profile(profile_step=1)
```

### 3.4 Expected Kernel Launch Pattern

For a single training step, the baseline implementation generates:

```
Per Transformer Layer (~17 kernel launches):
├── RMSNorm (pre-attention)         : 1 kernel
├── Q Projection                    : 1 kernel
├── K Projection                    : 1 kernel
├── V Projection                    : 1 kernel
├── RoPE (query)                    : 1 kernel
├── RoPE (key)                      : 1 kernel
├── Attention scores (QK^T)         : 1 kernel
├── Softmax                         : 1 kernel
├── Attention application (softmax*V): 1 kernel
├── Output Projection               : 1 kernel
├── Residual Add                    : 1 kernel
├── RMSNorm (pre-FFN)              : 1 kernel
├── Gate Projection                 : 1 kernel
├── Up Projection                   : 1 kernel
├── SiLU Activation                 : 1 kernel
├── Element-wise Multiply           : 1 kernel
└── Down Projection                 : 1 kernel

Total per step (4 layers): ~68 kernels (forward only)
With backward pass: ~136 kernels per step
```

**Optimization Implications:**
- High kernel launch overhead
- Many small operations
- Opportunities for fusion

### 3.5 Running the Baseline

**Quick Start:**

```bash
# Basic run without profiling
./run_baseline.sh

# Or manually
python3 tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 50
```

**With PyTorch Profiler:**

```bash
# Using helper script
./run_pytorch_profiler.sh

# Or manually
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --profile-dir ./pytorch_profiles \
    --profile-memory
```

**With DeepSpeed FLOPS Profiler:**

```bash
# Using helper script
./run_deepspeed_flops.sh

# Or manually
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-deepspeed-flops \
    --flops-profile-step 10
```

---

## 4. Exercise 1: Baseline Performance Analysis

### 4.1 Objective

Establish baseline performance metrics for Tiny LLaMA V1 and understand the profiling methodology that will be used throughout the workshop.

**What you'll learn:**
- How to run the baseline model
- How to enable and use PyTorch Profiler
- How to interpret basic profiling output
- What "good" performance looks like for this model
- How to identify top operations consuming time

### 4.2 Step-by-Step Instructions

#### Step 1: Run Baseline Training

First, let's run the basic model without any profiling to establish a clean baseline:

```bash
# Navigate to version1_pytorch_baseline directory
cd ~/castille-ai-workshop-training/version1_pytorch_baseline/

# Run basic training
python3 tiny_llama_v1.py --batch-size 8 --seq-len 128 --num-steps 20
```

**Expected Output:**

```
==========================================
Tiny LLaMA V1 - PyTorch Baseline
==========================================
Configuration:
  Batch Size: 8
  Sequence Length: 128
  Number of Steps: 20
  Hidden Dim: 256
  Num Layers: 4
  Num Heads: 8
  Intermediate Dim: 512

Model Configuration:
  Total Parameters: 2,345,984
  Model Size: 9.2 MB (FP32)

Initializing model and optimizer...
Using device: cuda
GPU: AMD Instinct MI250X

Starting training...
Step 1/20: Loss = 6.9088, Time = 0.234 seconds
Step 2/20: Loss = 6.9076, Time = 0.046 seconds
Step 3/20: Loss = 6.9065, Time = 0.045 seconds
Step 4/20: Loss = 6.9054, Time = 0.044 seconds
...
Step 20/20: Loss = 6.8821, Time = 0.044 seconds

==========================================
Performance Summary:
==========================================
Average time per step: 0.045 seconds
Training speed: 177.8 samples/sec
Peak memory usage: 2847 MB
Avg time per forward: 0.022 seconds
Avg time per backward: 0.018 seconds
Avg time per optimizer: 0.005 seconds
==========================================
```

**Record the following baseline metrics:**
- Training speed: _____ samples/sec
- Peak memory usage: _____ MB
- Avg time per step: _____ ms
- GPU name and memory

**Key Observations:**

1. **First iteration is slower:** Step 1 takes ~234ms vs ~44ms for subsequent steps
   - Reason: Kernel compilation, memory allocation, cache warming
   - **Always exclude first iteration from measurements**

2. **Consistent timing:** Steps 2-20 have similar timing
   - Good sign: stable performance
   - Small variance indicates consistent GPU utilization

3. **Memory usage:** ~2.8 GB for this configuration
   - Includes: Model weights (9 MB) + optimizer states (36 MB) + activations + gradients

#### Step 2: Enable PyTorch Profiler

Now let's add PyTorch profiler to understand what's happening under the hood:

```bash
# Run with PyTorch profiler enabled
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 20 \
    --enable-pytorch-profiler \
    --profile-dir ./exercise1_profiles \
    --profile-steps 5
```

**What this does:**
- Profiles steps 5-10 (after warmup)
- Records CPU and GPU operations
- Tracks memory allocations
- Generates TensorBoard-compatible traces

**Expected Output:**

```
==========================================
Tiny LLaMA V1 - PyTorch Baseline (Profiling Enabled)
==========================================
... (same as before) ...

Profiling enabled: Steps 5-10
Profile data will be saved to: ./exercise1_profiles/

Step 1/20: Loss = 6.9088, Time = 0.245 seconds
Step 2/20: Loss = 6.9076, Time = 0.048 seconds
Step 3/20: Loss = 6.9065, Time = 0.047 seconds
Step 4/20: Loss = 6.9054, Time = 0.046 seconds
Step 5/20: Loss = 6.9043, Time = 0.052 seconds [PROFILING]
Step 6/20: Loss = 6.9032, Time = 0.053 seconds [PROFILING]
...
Step 10/20: Loss = 6.8989, Time = 0.052 seconds [PROFILING]
Step 11/20: Loss = 6.8978, Time = 0.046 seconds
...

Profiling complete!
Profile files generated:
  - ./exercise1_profiles/trace_step_5_10.json
  - ./exercise1_profiles/events.out.tfevents.*
  - ./exercise1_profiles/performance_summary.json

Average time per step: 0.048 seconds (with profiling overhead)
Training speed: 166.7 samples/sec
Peak memory usage: 3124 MB
```

**Answer these questions in your results file:**

1. How much overhead did profiling add to training time?
   - Without profiling: ~0.045 seconds/step
   - With profiling: ~0.048-0.052 seconds/step
   - Overhead: ~6-15% (acceptable for profiling)

2. What files were generated in the `exercise1_profiles/` directory?

```bash
ls -lh ./exercise1_profiles/
```

3. What's the difference in memory usage with profiling enabled?
   - Extra memory needed for profiler data structures

#### Step 3: Analyze Profiling Results with TensorBoard

Launch TensorBoard to visualize the profiling results:

```bash
# Launch TensorBoard (run in background or separate terminal)
tensorboard --logdir ./exercise1_profiles --port 6006

# If TensorBoard is not available, examine JSON traces
# We'll show alternative analysis methods below
```

**TensorBoard Analysis:**

1. Open your browser to `http://localhost:6006` (or your server address)
2. Navigate to the "PROFILE" tab
3. Select the most recent run

**Explore the following views:**

**A. Overview Page:**

- **Performance Summary:** Shows step time breakdown
- **Run Environment:** GPU model, driver version, CUDA/ROCm version
- **Recommendation:** TensorBoard may suggest optimizations

**B. Trace Viewer:**

- Timeline of CPU and GPU operations
- Each row represents a thread or GPU stream
- Zoom in to see individual kernel launches
- Look for:
  - GPU idle time (gaps in GPU timeline)
  - CPU bottlenecks
  - Memory transfer operations

**C. Operator View:**

Shows aggregated statistics for each operation type:

```
Top Operations by Total Time:
Operation                          | Calls | GPU Time | CPU Time | Total Time
-----------------------------------|-------|----------|----------|------------
aten::mm (matrix multiply)         | 240   | 18.5 ms  | 0.2 ms   | 18.7 ms
aten::addmm (matrix multiply+add)  | 480   | 15.3 ms  | 0.3 ms   | 15.6 ms
aten::bmm (batch matrix multiply)  | 160   | 12.1 ms  | 0.1 ms   | 12.2 ms
aten::softmax                      | 80    | 8.4 ms   | 0.1 ms   | 8.5 ms
aten::mul (element-wise multiply)  | 320   | 3.2 ms   | 0.1 ms   | 3.3 ms
aten::add_ (in-place add)          | 160   | 2.8 ms   | 0.1 ms   | 2.9 ms
aten::silu (SiLU activation)       | 80    | 2.1 ms   | 0.1 ms   | 2.2 ms
aten::rsqrt (RMSNorm)              | 160   | 1.5 ms   | 0.1 ms   | 1.6 ms
```

**Document in your results file:**

**Top 3 longest-running operations:**
1. _________________
2. _________________
3. _________________

**D. Memory Timeline:**

- Shows memory allocation over time
- Peak memory during forward pass or backward pass?
- Memory spikes indicate large tensor allocations

**Document:**
- Peak memory: _____ MB
- When does peak occur: Forward / Backward / Optimizer
- Are there memory spikes? Yes / No

#### Step 4: Alternative Analysis (Without TensorBoard)

If TensorBoard is not available, analyze the JSON trace directly:

```bash
# View performance summary
cat ./exercise1_profiles/performance_summary.json | python3 -m json.tool
```

Use the Chrome trace viewer or analysis tools to identify the top operations by execution time. Look for patterns in:
- Matrix multiplication operations (mm, addmm, bmm)
- Attention-related kernels
- FFN operations
- Normalization operations

#### Step 5: Identify Performance Patterns

Based on your analysis, identify patterns in the baseline model:

**Check these patterns in your results:**

**Compute Patterns:**

- [ ] Matrix multiplications (mm, addmm, bmm) dominate compute time
- [ ] Attention operations consume ~35-45% of total time
- [ ] FFN operations consume ~30-40% of total time
- [ ] Many small operations with low individual utilization
- [ ] Kernel launch overhead visible in timeline

**Memory Patterns:**

- [ ] Memory usage grows during forward pass
- [ ] Peak memory during attention computation
- [ ] Gradient tensors allocated during backward pass
- [ ] Frequent small allocations for intermediate tensors

**Optimization Opportunities:**

Based on the profiling results, rank these optimizations by potential benefit:

- [ ] **High:** Kernel fusion (reduce number of operations)
- [ ] **High:** Fused QKV projection in attention
- [ ] **High:** Flash Attention implementation (reduce memory)
- [ ] **Medium:** Memory layout optimization
- [ ] **Medium:** Mixed precision training (FP16)
- [ ] **Low:** Batch size scaling (already reasonable)

### 4.3 Expected Results

After completing this exercise, you should have:

#### Performance Baseline

Representative ranges (actual results will vary by hardware):

- **Training Speed:** 50-200 samples/sec
- **GPU Utilization:** 60-75% (typical for baseline PyTorch)
- **Memory Usage:** 2-4 GB (depends on batch size)
- **Kernel Count:** 60-80 different kernel launches per step
- **MFU (estimated):** 20-35% (memory-bound workload)

#### Key Observations

1. **Attention operations consume ~35-45% of total compute time**
   - QKV projections: separate kernel launches
   - Attention computation: O(S^2) memory complexity
   - Softmax: memory-bound operation

2. **FFN operations consume ~30-40% of total time**
   - Gate/Up projections: separate operations
   - SwiGLU: sequential activation and multiplication

3. **Matrix multiplications (GEMM) are the dominant kernels**
   - Linear layers in projections
   - Attention score computation
   - Good candidates for optimization

4. **Multiple small operations create kernel launch overhead**
   - Element-wise operations (add, multiply, activation)
   - Normalization layers
   - Residual connections

5. **Memory allocation patterns show optimization opportunities**
   - Intermediate tensors in attention
   - Separate activations in FFN
   - Gradient storage

#### Profiling Data Generated

```
exercise1_profiles/
├── trace_step_5_10.json           # Chrome trace format
├── events.out.tfevents.*          # TensorBoard events
├── performance_summary.json       # Aggregated metrics
└── memory_timeline.json           # Memory usage over time
```

### 4.4 Troubleshooting

#### Common Issues

**1. CUDA/ROCm Memory Errors**

```bash
# Error: RuntimeError: CUDA out of memory
# Solution: Reduce batch size or sequence length
python3 tiny_llama_v1.py --batch-size 4 --seq-len 64 --num-steps 10
```

**2. Profiling Files Not Generated**

```bash
# Check permissions and disk space
ls -la ./exercise1_profiles/
df -h .

# Create directory manually
mkdir -p exercise1_profiles
chmod 755 exercise1_profiles
```

**3. TensorBoard Not Loading**

```bash
# Try different port
tensorboard --logdir ./exercise1_profiles --port 6007

# Check if port is in use
netstat -tuln | grep 6006

# Or examine JSON files directly (see alternative analysis above)
```

**4. Low GPU Utilization**

```bash
# Check if GPU is being used
rocm-smi

# Monitor GPU during training (in separate terminal)
watch -n 1 rocm-smi

# Check for CPU bottlenecks
htop
```

**5. Inconsistent Timing**

```bash
# Ensure no other processes are using GPU
rocm-smi

# Run with deterministic mode
python3 tiny_llama_v1.py --deterministic --seed 42
```

### 4.5 Analysis Questions

Answer these questions based on your results:

**1. What is the primary bottleneck in the baseline model?**
   - [ ] Memory bandwidth (many small operations)
   - [ ] Compute utilization (GPU not fully utilized)
   - [ ] Kernel launch overhead (too many launches)
   - [ ] Data loading (CPU bottleneck)

**Answer:** Likely a combination of memory bandwidth and kernel launch overhead. The baseline has many small operations that don't fully utilize the GPU.

**2. Which operations would benefit most from fusion?**
   - [ ] QKV projections in attention
   - [ ] Gate/Up projections in SwiGLU
   - [ ] Layer normalization operations
   - [ ] All of the above

**Answer:** All of the above. Version 2 will address these with kernel fusion.

**3. What percentage of time is spent in attention vs FFN?**

Based on profiling data:
- Attention: ~_____%
- FFN: ~_____%
- Other (norms, residuals): ~_____%

**4. Based on memory usage patterns, what optimization would help most?**
   - [ ] Gradient checkpointing (reduce activation memory)
   - [ ] Flash Attention (reduce attention memory from O(S^2) to O(S))
   - [ ] Mixed precision (reduce memory footprint by 2x)
   - [ ] Tensor fusion (reduce intermediate tensor allocations)

**Answer:** Flash Attention for long sequences, tensor fusion for overall efficiency.

### 4.6 Key Takeaways

**What We Learned:**

1. **Baseline performance characteristics:**
   - Training speed: _____ samples/sec (record your value)
   - GPU utilization: Moderate (60-75%)
   - Memory usage: Reasonable for batch size

2. **Primary bottlenecks identified:**
   - Separate kernel launches for QKV, Gate/Up projections
   - O(S^2) memory usage in attention
   - Memory bandwidth limitations

3. **Optimization targets for Version 2:**
   - QKV fusion (combine 3 operations into 1)
   - SwiGLU fusion (combine gate/up projections)
   - Custom fused kernels for common patterns

4. **Profiling methodology:**
   - PyTorch Profiler provides detailed operator-level insights
   - TensorBoard visualization helps identify patterns
   - JSON traces enable programmatic analysis

**Next Steps:**

- Document your findings
- Compare with expected results (are your metrics in the expected ranges?)
- Identify top 3 optimization targets for Version 2
- Save your profiling data for comparison with optimized versions

**Exercise Complete When:**

- [ ] Baseline training runs successfully
- [ ] Profiling data generated and analyzed
- [ ] Performance metrics documented
- [ ] Top operations identified
- [ ] Bottlenecks understood
- [ ] Ready to proceed to memory analysis

---

**Next Exercise:** [Exercise 2 - Memory Analysis & Optimization](#5-exercise-2-memory-analysis--optimization)

---

## 5. Exercise 2: Memory Analysis & Optimization

### 5.1 Objective

Understand memory usage patterns, identify memory bottlenecks, and analyze memory bandwidth utilization in the baseline Tiny LLaMA model.

**What you'll learn:**
- How memory scales with batch size and sequence length
- Where peak memory is consumed (forward, backward, optimizer)
- Memory bandwidth utilization patterns
- How to identify memory-bound vs compute-bound operations
- Memory optimization opportunities

### 5.2 Background: Why Memory Matters

Memory optimization is crucial for transformer models because:

**Memory Bandwidth:**
- Often the limiting factor, especially for small models
- Modern GPUs have very high compute (TFLOPS) but limited bandwidth (TB/s)
- Memory-bound operations don't fully utilize GPU compute

**Peak Memory:**
- Determines maximum batch size and model size
- Out-of-memory (OOM) errors are common
- Larger batches → better GPU utilization

**Memory Fragmentation:**
- Multiple small allocations reduce effective memory
- Garbage collection overhead
- Can cause OOM even with available memory

**Attention Memory:**
- Quadratic scaling: O(S^2) with sequence length
- Major bottleneck for long sequences
- Target for Flash Attention optimization

### 5.3 Step-by-Step Instructions

#### Step 1: Memory-Focused Profiling

Run profiling with enhanced memory analysis for different batch sizes:

```bash
# Batch size 4
python3 tiny_llama_v1.py \
    --batch-size 4 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --profile-memory \
    --profile-dir ./memory_analysis_bs4

# Batch size 8
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --profile-memory \
    --profile-dir ./memory_analysis_bs8

# Batch size 16
python3 tiny_llama_v1.py \
    --batch-size 16 \
    --seq-len 128 \
    --num-steps 15 \
    --enable-pytorch-profiler \
    --profile-memory \
    --profile-dir ./memory_analysis_bs16
```

**Expected Output for Each Run:**

```
==========================================
Tiny LLaMA V1 - Memory Profiling
==========================================
Configuration:
  Batch Size: 8
  Sequence Length: 128
  ...

Memory Profiling Enabled

Step 1/15: Loss = 6.9088, Time = 0.245 s, Memory = 2847 MB
...
Step 15/15: Loss = 6.8765, Time = 0.046 s, Memory = 2847 MB

==========================================
Memory Analysis Summary:
==========================================
Peak Memory Usage: 2847 MB
Average Memory Usage: 2654 MB
Memory at Forward Pass: 2123 MB
Memory at Backward Pass: 2847 MB
Memory at Optimizer Step: 2456 MB
Number of Allocations: 1234
Largest Tensor: 512 MB (attention_scores)
==========================================
```

**Record memory usage for each batch size in your results file:**

| Batch Size | Peak Memory (MB) | Avg Memory (MB) | Training Speed (samples/sec) |
|------------|------------------|-----------------|------------------------------|
| 4          | _______          | _______         | _______                      |
| 8          | _______          | _______         | _______                      |
| 16         | _______          | _______         | _______                      |

**Questions to Answer:**

1. **Memory Scaling:** Does memory double when batch size doubles?
   - If yes → Linear scaling (good)
   - If more than double → Superlinear scaling (fragmentation or inefficiency)

2. **Throughput Scaling:** Does throughput double when batch size doubles?
   - If yes → Perfect scaling
   - If less → Diminishing returns (memory bandwidth limit)

3. **Memory Efficiency:** What's the peak-to-average memory ratio?
   - High ratio → Memory spikes, potential for optimization
   - Low ratio → Consistent memory usage

#### Step 2: Memory Timeline Analysis

Analyze memory patterns using TensorBoard:

```bash
# Launch TensorBoard for memory analysis
tensorboard --logdir ./memory_analysis_bs8 --port 6007
```

**In TensorBoard:**

1. Go to the **PROFILE** tab
2. Select **Memory Viewer** or **Memory Timeline** view
3. Examine the memory usage pattern over time

**What to Look For:**

**A. Memory Allocation Pattern:**

```
Memory (MB)
    |
3000|                    ╱‾‾‾‾‾╲
    |                   /       \
2500|                  /         \___________
    |                 /
2000|        ╱‾‾‾‾‾‾╱
    |       /
1500|______/
    |
    +-----|-----|-----|-----|-----|------> Time
         Fwd  Attn  FFN  Bwd  Opt  Done
```

- **Forward pass:** Memory increases as activations are computed
- **Attention:** Often creates a spike (attention matrices)
- **FFN:** Additional activation memory
- **Backward pass:** Gradient tensors allocated
- **Optimizer:** Parameter updates

**B. Memory Peaks:**

Document when peak memory occurs:
- [ ] During forward pass (activations)
- [ ] During attention computation (attention matrices)
- [ ] During backward pass (gradients)
- [ ] During optimizer step (momentum buffers)

**C. Memory Deallocation:**

- Are there clear drops in memory usage?
- Does memory return to baseline after each step?
- Are tensors being deallocated promptly?

**Record in your results file:**

**Memory Pattern Analysis:**
- Peak memory occurs at: _______________________
- Largest memory spike caused by: _______________________
- Memory is deallocated: Promptly / Delayed / Not at all
- Memory usage pattern: Steady / Fluctuating / Spiking

#### Step 3: Sequence Length Scaling

Test how memory scales with sequence length:

```bash
# Sequence length 64
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 64 \
    --num-steps 10 \
    --profile-memory \
    --profile-dir ./memory_seq64

# Sequence length 128 (baseline)
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 10 \
    --profile-memory \
    --profile-dir ./memory_seq128

# Sequence length 256
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 256 \
    --num-steps 10 \
    --profile-memory \
    --profile-dir ./memory_seq256

# Sequence length 512 (might OOM - use smaller batch if needed)
python3 tiny_llama_v1.py \
    --batch-size 4 \
    --seq-len 512 \
    --num-steps 5 \
    --profile-memory \
    --profile-dir ./memory_seq512
```

**Record sequence length scaling:**

| Seq Length | Batch Size | Peak Memory (MB) | Memory Increase | Scaling Factor |
|------------|------------|------------------|-----------------|----------------|
| 64         | 8          | _______          | baseline        | 1.0x           |
| 128        | 8          | _______          | _______         | _______        |
| 256        | 8          | _______          | _______         | _______        |
| 512        | 4          | _______          | _______         | _______        |

**Memory Scaling Analysis:**

Calculate the scaling factor:
```
Scaling Factor = Memory(S) / Memory(S_baseline)

For attention memory (theoretical):
- Linear components: O(S) → 2x when S doubles
- Attention matrix: O(S^2) → 4x when S doubles

Expected combined: ~3x when S doubles (for attention-heavy workloads)
```

**Answer these questions:**

1. **What is the memory scaling pattern?**
   - [ ] Linear (~2x when sequence doubles)
   - [ ] Quadratic (~4x when sequence doubles)
   - [ ] Between linear and quadratic (~3x)

2. **Which component shows steepest memory scaling?**
   - Run separate profiling focusing on attention vs FFN
   - Check memory timeline for attention layers

3. **At what sequence length do you hit memory limits?**
   - Record the maximum sequence length before OOM
   - Note the batch size at that limit

#### Step 4: Identifying Memory Hotspots

Use profiling to identify which operations consume most memory:

```bash
# Run with detailed operator profiling
python3 tiny_llama_v1.py \
    --batch-size 8 \
    --seq-len 128 \
    --num-steps 10 \
    --enable-pytorch-profiler \
    --profile-memory \
    --profile-operators \
    --profile-dir ./memory_hotspots
```

**Analyze the operator memory usage:**

Review the memory profiling output and trace files to identify operators with highest memory allocation. Use the PyTorch Profiler's memory view or trace analysis to examine memory allocation patterns.

**Record top memory-consuming operations:**

1. _________________: _______ MB
2. _________________: _______ MB
3. _________________: _______ MB
4. _________________: _______ MB
5. _________________: _______ MB

**Common Memory Hotspots:**

- **Attention scores:** `[B, H, S, S]` matrices (quadratic in S)
- **Query/Key/Value states:** `[B, S, D]` tensors
- **FFN intermediate:** `[B, S, D_ff]` tensors
- **Gradients:** Same size as parameters + activations

#### Step 5: Memory Bandwidth Analysis

Analyze memory bandwidth utilization:

**Calculate memory bandwidth manually:**

For batch_size=8, seq_len=128, hidden_dim=256, n_layers=4:

1. **Estimate memory traffic per step:**
   - Forward pass: QKV weights + activations + FFN weights
   - Backward pass: ~2× forward pass
   - Total: Depends on model size and batch configuration

2. **Calculate bandwidth utilization:**
   - Memory bandwidth = Total memory traffic / Step time
   - Compare with theoretical peak (e.g., MI250X: ~1.6 TB/s per GCD)
   - Utilization % = (Actual bandwidth / Peak bandwidth) × 100

3. **Calculate arithmetic intensity:**
   - Arithmetic intensity = FLOPs / Memory traffic (bytes)
   - < 10 FLOPS/byte: Memory-bound
   - > 100 FLOPS/byte: Compute-bound
   - 10-100 FLOPS/byte: Mixed workload

Record your observations based on the profiling data collected.

**Record in your results file:**

**Bandwidth Analysis:**
- Memory Traffic per Step: _______ GB
- Memory Bandwidth Used: _______ GB/s
- Theoretical Peak Bandwidth: _______ GB/s
- Bandwidth Utilization: _______%
- Arithmetic Intensity: _______ FLOPS/byte
- Workload Classification: _______

### 5.4 Analysis and Interpretation

#### Memory Scaling Patterns

**Batch Size Scaling:**

Expected pattern:
- Memory ≈ Base + (Batch_size × Per_sample_memory)
- Should be approximately linear
- If superlinear → fragmentation or inefficiency

**Sequence Length Scaling:**

Components:
- Linear: Activations, most projections
- Quadratic: Attention matrices `[B, H, S, S]`
- Combined: Between linear and quadratic

**Typical Results:**

| Component      | S=64 | S=128 | S=256 | Scaling |
|----------------|------|-------|-------|---------|
| Parameters     | 9MB  | 9MB   | 9MB   | O(1)    |
| Activations    | ~1GB | ~2GB  | ~4GB  | O(S)    |
| Attention      | ~100MB | ~400MB | ~1.6GB | O(S^2) |
| Total          | ~1.1GB | ~2.4GB | ~5.6GB | Mixed  |

#### Memory Bottleneck Classification

**Workload Type Determination:**

```
Arithmetic Intensity (FLOPS/byte):
- < 10: Memory-bound (bandwidth limited)
- 10-100: Mixed workload
- > 100: Compute-bound (ALU limited)

Typical Transformer Training: 20-50 FLOPS/byte (mixed, leaning memory-bound)
```

**Signs of Memory-Bound Workload:**
- Low GPU compute utilization (<70%)
- High memory bandwidth utilization (>60%)
- Many small operations
- Frequent memory transfers

**Signs of Compute-Bound Workload:**
- High GPU compute utilization (>80%)
- Low memory bandwidth utilization (<50%)
- Large matrix multiplications dominate
- Good arithmetic intensity

### 5.5 Memory Optimization Opportunities

Based on your analysis, rank these optimizations:

**1. Flash Attention**
- **Impact:** Reduces attention memory from O(S^2) to O(S)
- **Benefit:** Enables much longer sequences
- **When:** Always beneficial for S > 512
- **Rank:** _____ (1-4)

**2. Gradient Checkpointing**
- **Impact:** Trades compute for memory (recompute activations)
- **Benefit:** Reduces activation memory by ~2-4x
- **When:** Memory-constrained, willing to sacrifice 20-30% speed
- **Rank:** _____ (1-4)

**3. Mixed Precision (FP16/BF16)**
- **Impact:** Reduces memory per parameter by 2x
- **Benefit:** Allows 2x larger batch or model
- **When:** Always beneficial if hardware supports it
- **Rank:** _____ (1-4)

**4. Kernel Fusion**
- **Impact:** Reduces intermediate tensor allocations
- **Benefit:** Lower memory footprint, less fragmentation
- **When:** Many small operations (already the case)
- **Rank:** _____ (1-4)

### 5.6 Expected Results

After completing this exercise, you should have:

**Memory Usage Baseline:**
- Peak memory: 2-4 GB (batch_size=8, seq_len=128)
- Memory scaling: ~Linear with batch size, ~Quadratic with sequence
- Memory hotspots: Attention matrices, FFN intermediate tensors
- Bandwidth utilization: 30-60% (memory-bound to mixed)

**Key Findings:**

1. **Attention Memory Dominates for Long Sequences**
   - At S=512, attention alone can consume GBs
   - Quadratic scaling makes long sequences expensive
   - Flash Attention is critical optimization target

2. **Memory Fragmentation Observable**
   - Peak-to-average ratio often 1.2-1.5x
   - Many small allocations create overhead
   - Tensor fusion can reduce fragmentation

3. **Bandwidth Utilization is Moderate**
   - Typically 30-60% for baseline PyTorch
   - Room for improvement through fusion
   - Memory-bound operations limit performance

4. **Linear Components Well-Behaved**
   - FFN and most projections scale linearly
   - Predictable memory requirements
   - Batch size scaling is efficient

### 5.7 Troubleshooting

**Out of Memory Errors:**

```bash
# Error: RuntimeError: CUDA out of memory
# Solution 1: Reduce batch size
python3 tiny_llama_v1.py --batch-size 2 --seq-len 128

# Solution 2: Reduce sequence length
python3 tiny_llama_v1.py --batch-size 8 --seq-len 64

# Solution 3: Enable gradient accumulation (if implemented)
python3 tiny_llama_v1.py --batch-size 4 --gradient-accumulation-steps 2
```

**Memory Profiling Overhead:**

```bash
# If profiling causes OOM, reduce profiling frequency
python3 tiny_llama_v1.py --profile-steps 2  # Profile fewer steps
```

**Memory Fragmentation:**

```bash
# Set memory allocator configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or use expandable segments (PyTorch 2.0+)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 5.8 Analysis Questions

Answer these questions based on your results:

**1. What is the memory scaling behavior?**
   - Batch size scaling: [ ] Linear [ ] Superlinear [ ] Sublinear
   - Sequence length scaling: [ ] Linear [ ] Quadratic [ ] Cubic

**2. Where is peak memory consumed?**
   - [ ] Forward pass (activations)
   - [ ] Backward pass (gradients)
   - [ ] Optimizer step (parameter updates)
   - [ ] Attention computation (attention matrices)

**3. What is the primary memory optimization target?**
   - [ ] Reduce attention memory (Flash Attention)
   - [ ] Reduce activation memory (checkpointing)
   - [ ] Reduce parameter memory (mixed precision)
   - [ ] Reduce fragmentation (kernel fusion)

**4. Is the workload memory-bound or compute-bound?**
   - [ ] Memory-bound (low arithmetic intensity, <10 FLOPS/byte)
   - [ ] Compute-bound (high arithmetic intensity, >100 FLOPS/byte)
   - [ ] Mixed (moderate arithmetic intensity, 10-100 FLOPS/byte)

**5. What memory optimization would provide the biggest benefit?**

Rank by expected impact:
1. _______________________________________
2. _______________________________________
3. _______________________________________
4. _______________________________________

### 5.9 Key Takeaways

**What We Learned:**

1. **Memory Scaling Patterns:**
   - Batch size: Linear (good)
   - Sequence length: Between linear and quadratic (attention dominates)
   - Peak memory occurs during backward pass or attention computation

2. **Memory Bottlenecks Identified:**
   - Attention matrices: O(S^2) memory usage
   - Intermediate tensors: FFN activations
   - Memory fragmentation from many small allocations

3. **Bandwidth Utilization:**
   - Moderate utilization (30-60%) indicates mixed workload
   - Room for optimization through kernel fusion
   - Memory bandwidth limits throughput for small models

4. **Optimization Priorities:**
   - Flash Attention: Critical for long sequences (S > 512)
   - Kernel fusion: Reduces fragmentation and bandwidth pressure
   - Mixed precision: 2x memory reduction, always beneficial

**Next Steps:**

- Document memory analysis in results file
- Compare memory patterns across configurations
- Identify top 3 memory optimization targets
- Understand memory-compute trade-offs
- Proceed to Exercise 3 for bottleneck identification

**Exercise Complete When:**

- [ ] Memory profiling completed for multiple batch sizes
- [ ] Sequence length scaling analyzed
- [ ] Memory hotspots identified
- [ ] Bandwidth utilization calculated
- [ ] Optimization priorities ranked
- [ ] Ready to proceed to bottleneck identification

---

## 6. Exercise 3: Performance Study Across Problem Sizes

### 6.1 Objective

Learn how model performance scales with different problem sizes by using the automated performance study launcher. This exercise demonstrates:

- How performance varies across tiny to very large model configurations
- Scaling characteristics of attention and FFN operations
- Memory and compute requirements for different model sizes
- How to establish performance baselines for optimization comparisons

**Time Required:** 15-30 minutes (depending on problem sizes tested)

### 6.2 Understanding the Performance Study Script

The `launch_performance_study.sh` script provides pre-configured problem sizes:

| Size | Hidden Dim | Layers | Seq Len | Batch | Params | Expected Time |
|------|-----------|--------|---------|-------|--------|---------------|
| **tiny** | 256 | 4 | 128 | 8 | ~2.9M | <5s/iter |
| **small** | 512 | 8 | 256 | 8 | ~20.9M | 10-30s/iter |
| **medium** | 1024 | 12 | 512 | 16 | ~167M | 30-60s/iter |
| **large** | 2048 | 16 | 1024 | 8 | ~1.3B | 1-3min/iter |
| **very_large** | 4096 | 24 | 2048 | 4 | ~10.7B | 5-10min/iter |

**Script Features:**
- Automatic configuration based on problem size
- Output organization with timestamps
- Configuration metadata in JSON format
- Optional profiler integration
- Performance metrics extraction
- Next steps guidance

### 6.3 Step-by-Step Instructions

#### Step 1: Run Tiny Problem Size (Quick Validation)

Start with the smallest size to verify everything works:

```bash
cd ~/castille-ai-workshop-training/version1_pytorch_baseline/

# Run tiny problem size (fast validation)
./launch_performance_study.sh tiny
```

**Expected Output:**
```
================================================================================
CASTILLE AI WORKSHOP - VERSION 1 BASELINE PERFORMANCE STUDY
================================================================================

Problem Size: TINY
Configuration:
  Hidden Dimension:    256
  Number of Layers:    4
  Sequence Length:     128
  Batch Size:          8
  Training Steps:      50
  Est. Parameters:     ~2.9M
  Expected Time:       <5s/iter
  Profilers Enabled:   false

Output Directory: performance_results_tiny_20251014_123456
================================================================================

Starting V1 Baseline training...
...
================================================================================
PERFORMANCE STUDY COMPLETE
================================================================================
Total Runtime: 42s
Throughput: 95.2 samples/sec
Peak Memory: 342 MB
```

**Observe:**
- Quick completion time
- Low memory usage
- Baseline throughput metrics

#### Step 2: Run Medium Problem Size (Workshop Standard)

Test the standard workshop configuration:

```bash
# Run medium problem size with profiling enabled
./launch_performance_study.sh medium --enable-profilers
```

**Note:** This will take longer (5-10 minutes) due to profiling overhead.

**Expected Characteristics:**
- Longer runtime per iteration
- Higher memory usage
- More realistic model size for workshops
- Profiling data generated for analysis

#### Step 3: Compare Problem Sizes

Run multiple sizes to observe scaling:

```bash
# Run small size
./launch_performance_study.sh small

# Run medium size (if not done in Step 2)
./launch_performance_study.sh medium

# Optional: Run large (if you have time and memory)
# WARNING: This requires significant GPU memory (>16GB)
# ./launch_performance_study.sh large
```

#### Step 4: Analyze Results

Each run creates a timestamped output directory. Examine the results:

```bash
# List all performance study results
ls -lt performance_results_*/

# View latest tiny run configuration
cat performance_results_tiny_*/config.json

# View training output
cat performance_results_tiny_*/training_output.log

# Compare throughput across sizes
echo "=== Throughput Comparison ==="
for dir in performance_results_*/; do
    size=$(basename "$dir" | cut -d'_' -f3)
    throughput=$(grep "Throughput:" "$dir/training_output.log" | tail -1 | awk '{print $2, $3}')
    echo "$size: $throughput"
done

# Compare memory usage
echo ""
echo "=== Memory Usage Comparison ==="
for dir in performance_results_*/; do
    size=$(basename "$dir" | cut -d'_' -f3)
    memory=$(grep "Peak memory usage:" "$dir/training_output.log" | tail -1 | awk '{print $4, $5}')
    echo "$size: $memory"
done
```

#### Step 5: Record Scaling Observations

Create a comparison table from your results:

**Performance Scaling:**

| Problem Size | Parameters | Throughput (samples/s) | Memory (MB) | Time/Iter (s) |
|--------------|-----------|------------------------|-------------|---------------|
| tiny         | ~2.9M     | _________              | _________   | _________     |
| small        | ~20.9M    | _________              | _________   | _________     |
| medium       | ~167M     | _________              | _________   | _________     |

**Scaling Analysis:**

1. **Throughput Scaling:**
   - Does throughput decrease linearly with model size?
   - At what size does GPU become saturated?
   - How does batch size affect throughput?

2. **Memory Scaling:**
   - Is memory scaling proportional to parameter count?
   - Where does attention memory become significant?
   - What's the memory overhead ratio?

3. **Compute Characteristics:**
   - Which size achieves best GPU utilization?
   - How does arithmetic intensity change?
   - Is the workload memory-bound or compute-bound?

### 6.4 Understanding Scaling Patterns

**Expected Scaling Behavior:**

**1. Parameter Count Scaling:**
- Linear layers: Scale with D² (hidden dimension squared)
- Attention: Scales with D² for projections, S² for computation
- FFN: Scales with D × D_ff (typically D × 4D)

**2. Memory Scaling:**
- Parameters: Linear with model size
- Activations: Linear with batch size, quadratic with sequence length
- Peak memory: Dominated by activations for large sequences

**3. Compute Scaling:**
- FLOPs: Proportional to parameters × sequence length × batch size
- Time per iteration: Depends on GPU utilization
- Throughput: Inversely related to FLOPs per sample

**4. GPU Utilization:**
- Small models: Memory-bound, low GPU utilization
- Medium models: Mixed workload, moderate utilization
- Large models: Compute-bound, high GPU utilization

### 6.5 Expected Results

After completing this exercise, you should observe:

**Tiny → Small Transition (2.9M → 20.9M):**
- Parameter increase: ~7x
- Memory increase: ~5-8x
- Throughput decrease: ~3-5x
- GPU utilization: Still relatively low

**Small → Medium Transition (20.9M → 167M):**
- Parameter increase: ~8x
- Memory increase: ~6-10x (sequence length doubles!)
- Throughput decrease: ~5-10x
- GPU utilization: Significantly improved

**Key Observations:**

1. **Quadratic Attention Cost Visible:**
   - Medium (seq_len=512) shows significant attention overhead vs small (seq_len=256)
   - Memory increases faster than linear due to S² term
   - This motivates Flash Attention optimization

2. **Batch Size Impact:**
   - Medium uses batch_size=16 vs 8 for small/large
   - Better GPU utilization with larger batches
   - Memory-throughput trade-off visible

3. **Memory Becomes Limiting:**
   - Large/very_large reduce batch size to fit in memory
   - Attention matrices consume significant memory at long sequences
   - Gradient checkpointing would be beneficial

4. **Compute Patterns:**
   - Larger models approach compute-bound regime
   - Better GPU utilization percentage
   - GEMM operations dominate more clearly

### 6.6 Profiling Analysis (If Enabled)

If you ran with `--enable-profilers`, analyze the generated profiles:

```bash
# Navigate to profiled run
cd performance_results_medium_*/

# View performance summary
cat performance_summary.json | python3 -m json.tool

# Check for profiler outputs
ls -lh pytorch_profiles/
```

**Compare profiling results across sizes:**
- How does kernel distribution change?
- Which operations dominate in small vs large models?
- How does memory bandwidth utilization scale?

### 6.7 Troubleshooting

**Out of Memory Error:**

```bash
# Error: RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB

# Solution 1: Try the next smaller size
./launch_performance_study.sh small  # instead of medium

# Solution 2: Skip large/very_large on limited hardware
# These sizes require >16GB GPU memory
```

**Slow Execution:**

```bash
# If profiling is too slow, disable it
./launch_performance_study.sh medium  # without --enable-profilers

# Reduce number of steps for faster results (edit script or run directly)
python tiny_llama_v1.py --hidden-dim 1024 --num-layers 12 --seq-len 512 \
    --batch-size 16 --num-steps 20  # Reduced from 100
```

**Script Permission Denied:**

```bash
# Make script executable
chmod +x launch_performance_study.sh

# Then run
./launch_performance_study.sh tiny
```

### 6.8 Analysis Questions

Answer these based on your performance study results:

**1. Scaling Characteristics:**

Q: How does throughput scale with model size?
A: _________________________________________________________________

Q: At what model size does GPU utilization peak?
A: _________________________________________________________________

Q: Which component (attention vs FFN) dominates compute time?
A: _________________________________________________________________

**2. Memory Patterns:**

Q: How does memory scale with sequence length? (linear, quadratic, other?)
A: _________________________________________________________________

Q: What is the memory overhead ratio (peak / parameters)?
A: _________________________________________________________________

Q: At what point does attention memory become significant?
A: _________________________________________________________________

**3. Performance Optimization:**

Q: Which model size would benefit most from Flash Attention?
A: _________________________________________________________________

Q: Which size is most memory-bound vs compute-bound?
A: _________________________________________________________________

Q: What batch size would you recommend for medium model?
A: _________________________________________________________________

**4. Practical Insights:**

Q: What's the largest model you can train on your GPU?
A: _________________________________________________________________

Q: How would you improve throughput for the medium model?
A: _________________________________________________________________

Q: What's the optimal problem size for this workshop?
A: _________________________________________________________________

### 6.9 Key Takeaways

**1. Problem Size Dramatically Affects Performance:**
- Small models: Memory-bound, low GPU utilization
- Large models: Compute-bound, high GPU utilization
- Medium models: Sweet spot for learning optimizations

**2. Attention Memory Scales Quadratically:**
- Visible impact when comparing seq_len=256 vs 512 vs 1024
- Flash Attention is critical for long sequences
- Memory becomes limiting factor before compute

**3. Batch Size is a Key Tuning Parameter:**
- Larger batches improve GPU utilization
- Memory constraints force smaller batches for large models
- Trade-off between throughput and memory usage

**4. Automated Testing is Valuable:**
- Pre-configured sizes reduce manual configuration errors
- Consistent testing methodology across problem sizes
- Easy to reproduce and compare results

**5. Scaling Informs Optimization Strategy:**
- Tiny models: Not worth optimizing (I/O bound)
- Small-medium: Kernel fusion, mixed precision beneficial
- Large: Flash Attention, gradient checkpointing critical

**Next Steps:**

- Review all performance study results
- Document scaling patterns in your notes
- Identify which optimizations would have most impact
- Use baseline results to measure optimization improvements
- Proceed to comparative analysis with optimized versions

**Exercise Complete When:**

- [ ] At least 2 problem sizes tested (tiny + one other)
- [ ] Scaling patterns documented
- [ ] Memory and throughput metrics recorded
- [ ] Performance characteristics understood
- [ ] Optimization priorities identified
- [ ] Ready to compare with optimized versions

---

**Next Exercise:** Exercise 4 - Comparative Analysis with Optimized Versions

---

