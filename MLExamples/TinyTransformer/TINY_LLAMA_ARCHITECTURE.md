
# Tiny LLaMA Model Architecture and Mathematical Foundations

TINY_LLAMA_ARCHITECTURE.md from `HPCTrainingExamples/MLExamples/TinyTransformer` in the Training Examples repository

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Model Architecture](#model-architecture)
4. [Implementation Patterns](#implementation-patterns)
5. [Computational Complexity](#computational-complexity)
6. [Performance Characteristics](#performance-characteristics)

## Overview

Tiny LLaMA is a scaled-down implementation of the LLaMA (Large Language Model Meta AI) architecture, designed specifically for educational purposes and performance profiling workshops. This document provides comprehensive technical details of the model architecture, mathematical formulations, and implementation patterns used throughout the workshop versions.

### Key Design Principles

- **Educational Focus**: Simplified scale for understanding optimization techniques
- **Performance Profiling**: Designed to showcase bottlenecks and optimization opportunities
- **Progressive Optimization**: Architecture supports incremental improvements across workshop versions
- **Hardware Awareness**: Implementation patterns optimized for AMD ROCm and modern GPU architectures

## Mathematical Foundations

### 1. Transformer Architecture Core Equations

The Tiny LLaMA model follows the standard transformer decoder architecture with the following mathematical formulation:

#### Input Embedding and Positional Encoding

$$E = \text{Embedding}(x) \in \mathbb{R}^{B \times S \times d}$$

Where:
- $B$ = batch size
- $S$ = sequence length
- $d$ = model dimension (hidden_dim)
- $x \in \mathbb{Z}^{B \times S}$ = input token indices

#### RMSNorm (Root Mean Square Normalization)

Unlike standard LayerNorm, Tiny LLaMA uses RMSNorm for improved numerical stability:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

**Mathematical Advantages:**
- Eliminates mean centering computation
- Reduces numerical precision requirements
- Better gradient flow properties
- Simplified hardware implementation

#### Rotary Position Embeddings (RoPE)

Tiny LLaMA uses RoPE for position encoding, which rotates query and key vectors:

$$\begin{aligned}
q_m &= \text{RoPE}(q, m) = q \cdot e^{im\theta} \\
k_n &= \text{RoPE}(k, n) = k \cdot e^{in\theta}
\end{aligned}$$

Where:
- $m, n$ are position indices
- $\theta_j = 10000^{-2j/d}$ for dimension $j$
- Complex multiplication implemented as 2D rotation matrix

**RoPE Implementation:**

$$\begin{pmatrix}
q_{2i} \\
q_{2i+1}
\end{pmatrix} = \begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix} \begin{pmatrix}
q_{2i} \\
q_{2i+1}
\end{pmatrix}$$

### 2. Multi-Head Attention Mechanism

#### Standard Attention Formulation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### Multi-Head Attention

$$\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}$$

**Parameter Matrices:**
- $W_i^Q \in \mathbb{R}^{d \times d_k}$ (Query projection)
- $W_i^K \in \mathbb{R}^{d \times d_k}$ (Key projection)
- $W_i^V \in \mathbb{R}^{d \times d_v}$ (Value projection)
- $W^O \in \mathbb{R}^{hd_v \times d}$ (Output projection)

Where $d_k = d_v = d/h$ and $h$ is the number of attention heads.

#### Causal Attention Mask

For autoregressive generation, attention is masked to prevent looking ahead:

$$\text{mask}_{i,j} = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}$$

### 3. SwiGLU Feed-Forward Network

Tiny LLaMA uses SwiGLU activation instead of standard ReLU:

$$\begin{aligned}
\text{SwiGLU}(x) &= \text{Swish}(xW_1 + b_1) \odot (xW_2 + b_2) \\
\text{Swish}(x) &= x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}
\end{aligned}$$

**Complete FFN Block:**

$$\begin{aligned}
\text{FFN}(x) &= \text{SwiGLU}(x)W_3 + b_3 \\
&= (\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}
\end{aligned}$$

**Mathematical Properties:**
- Non-monotonic activation function
- Better gradient flow than ReLU
- Gating mechanism similar to LSTM gates
- Improved expressivity for language modeling

## Model Architecture

### 1. Complete Model Structure

```
Input Tokens → Embedding → [Transformer Block] × N → RMSNorm → Output Projection
```

### 2. Transformer Block Architecture

Each transformer block consists of:

```
x → RMSNorm → Multi-Head Attention → Residual Connection →
    RMSNorm → SwiGLU FFN → Residual Connection → Output
```

**Mathematical Formulation:**

$$\begin{aligned}
\text{attn\_out} &= x + \text{MultiHead}(\text{RMSNorm}(x)) \\
\text{block\_out} &= \text{attn\_out} + \text{FFN}(\text{RMSNorm}(\text{attn\_out}))
\end{aligned}$$

### 3. Model Dimensions and Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 1000 | Vocabulary size (reduced for workshop) |
| `hidden_dim` | 256 | Model dimension ($d$) |
| `n_layers` | 4 | Number of transformer blocks |
| `n_heads` | 8 | Number of attention heads |
| `n_kv_heads` | 4 | Number of key-value heads (GQA) |
| `intermediate_dim` | 512 | FFN intermediate dimension |
| `max_seq_len` | 128 | Maximum sequence length |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `norm_eps` | 1e-6 | RMSNorm epsilon |

### 4. Grouped-Query Attention (GQA)

Tiny LLaMA implements Grouped-Query Attention for memory efficiency:

$$\begin{aligned}
\text{n\_groups} &= \frac{\text{n\_heads}}{\text{n\_kv\_heads}} \\
\text{head\_dim} &= \frac{\text{hidden\_dim}}{\text{n\_heads}}
\end{aligned}$$

**Memory Reduction:**
- Reduces KV cache memory by factor of `n_groups`
- Maintains model quality while improving inference efficiency
- Critical for long sequence generation

## Implementation Patterns

### 1. PyTorch Module Hierarchy

```python
TinyLlama
├── embed_tokens: nn.Embedding
├── layers: nn.ModuleList[TransformerBlock]
│   ├── attention_norm: RMSNorm
│   ├── attention: Attention
│   │   ├── q_proj: nn.Linear
│   │   ├── k_proj: nn.Linear
│   │   ├── v_proj: nn.Linear
│   │   └── o_proj: nn.Linear
│   ├── ffn_norm: RMSNorm
│   └── mlp: MLP
│       ├── gate_proj: nn.Linear
│       ├── up_proj: nn.Linear
│       └── down_proj: nn.Linear
├── norm: RMSNorm
└── lm_head: nn.Linear
```

### 2. Forward Pass Computational Graph

**Memory Layout:**
```
Input: [batch_size, seq_len] → [batch_size, seq_len, hidden_dim]
Attention: [batch_size, seq_len, hidden_dim] → [batch_size, seq_len, hidden_dim]
FFN: [batch_size, seq_len, hidden_dim] → [batch_size, seq_len, hidden_dim]
Output: [batch_size, seq_len, hidden_dim] → [batch_size, seq_len, vocab_size]
```

**Key Implementation Details:**
- Pre-norm architecture (normalization before sub-layers)
- Residual connections around each sub-layer
- Weight sharing between embedding and output projection
- Causal masking for autoregressive generation

### 3. Memory Allocation Patterns

**Parameter Memory:**

$$\begin{aligned}
\text{Embedding} &: V \times d \\
\text{Attention} &: 4 \times L \times d \times d_{\text{head}} \times h \\
\text{FFN} &: L \times (2 \times d \times d_{\text{ff}} + d_{\text{ff}} \times d) \\
\text{LayerNorm} &: 2 \times L \times d + d \\
\text{Total} &\approx L \times d \times (4d + 3d_{\text{ff}}) + V \times d
\end{aligned}$$

**Activation Memory (per layer):**

$$\begin{aligned}
\text{Attention} &: B \times S \times d + B \times h \times S \times S \\
\text{FFN} &: B \times S \times d_{\text{ff}} \\
\text{KV Cache} &: 2 \times B \times S \times h_{\text{kv}} \times d_{\text{head}}
\end{aligned}$$

## Computational Complexity

### 1. Attention Complexity

**Standard Attention:**
- Time: $O(S^{2} \cdot d)$ per layer
- Memory: $O(S^{2})$ for attention matrix storage

**With Sequence Length $S = 128$:**
- Attention matrix: $128 \times 128 = 16,384$ elements per head
- Total attention matrices: $8 \times 16,384 = 131,072$ elements per layer

### 2. FFN Complexity

**Per Layer:**
- Time: $O(S \cdot d \cdot d_{\text{ff}})$
- Parameters: $2 \times d \times d_{\text{ff}} + d_{\text{ff}} \times d = 3 \times d \times d_{\text{ff}}$

**For Tiny LLaMA:**
- FFN operations: $S \times d \times d_{\text{ff}} = 128 \times 256 \times 512 = 16,777,216$ operations per layer

### 3. Total Model Complexity

**FLOP Count per Forward Pass:**

$$\begin{aligned}
\text{Embedding} &: B \times S \times d \\
\text{Attention} &: L \times B \times S \times (4 \times S \times d + S^{2} \times h) \\
\text{FFN} &: L \times B \times S \times 3 \times d \times d_{\text{ff}} \\
\text{LayerNorm} &: L \times B \times S \times d \times 2 \\
\text{Total} &\approx L \times B \times S \times (S \times d \times h + 3 \times d \times d_{\text{ff}})
\end{aligned}$$

**For Default Configuration:**
- Attention: $4 \times 128 \times 128 \times 256 \times 8 = 134,217,728$ FLOPs per layer
- FFN: $4 \times 128 \times 3 \times 256 \times 512 = 201,326,592$ FLOPs per layer
- **Total per forward pass**: ~1.34 GFLOPs (batch_size=1)

## Performance Characteristics

### 1. Memory Bandwidth Requirements

**Parameter Access:**
- Model parameters: ~2.8M parameters × 4 bytes = 11.2 MB
- Bandwidth requirement: 11.2 MB per forward pass

**Activation Memory:**
- Peak activation memory: ~50 MB (batch_size=8, seq_len=128)
- Memory bandwidth utilization: Critical for performance

### 2. Arithmetic Intensity

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

**For Tiny LLaMA:**
- FLOPs per forward pass: ~1.34 × 10^9
- Memory accessed: ~61.2 MB
- **Arithmetic Intensity**: ~21.9 FLOPs/byte

**Performance Implications:**
- Compute-bound on modern GPUs (good for optimization)
- Benefits significantly from kernel fusion
- Memory layout optimization crucial for performance

### 3. Optimization Opportunities

**Identified Bottlenecks:**
1. **Attention Memory**: $O(S^{2})$ memory scaling
2. **Kernel Launch Overhead**: Multiple small operations
3. **Memory Bandwidth**: Activation tensor transfers
4. **Load Imbalance**: Variable sequence lengths

**Optimization Strategies:**
1. **Kernel Fusion**: Combine multiple operations
2. **Flash Attention**: Reduce memory complexity
3. **Custom Kernels**: Triton implementations
4. **Memory Layout**: Optimize tensor arrangements

---

## References and Further Reading

1. **LLaMA Architecture**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
2. **RMSNorm**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
3. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
4. **SwiGLU**: "GLU Variants Improve Transformer" (Shazeer, 2020)
5. **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
6. **Grouped-Query Attention**: "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)

## Workshop Integration

This architecture serves as the foundation for all four workshop versions:

- **Version 1**: Standard PyTorch implementation
- **Version 2**: Kernel fusion optimizations
- **Version 3**: Custom Triton kernels
- **Version 4**: Ultra-fused implementations

Each version maintains this core architecture while progressively optimizing the implementation for maximum performance on AMD ROCm hardware.


