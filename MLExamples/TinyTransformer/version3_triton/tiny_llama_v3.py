#!/usr/bin/env python3
"""
Version 3: Tiny LLaMA with Triton Kernel Integration
Castille AI Workshop - AI Workload Profiling and Optimization

This version demonstrates custom Triton GPU kernels for critical operations,
achieving further performance improvements through low-level optimization.

Expected Performance:
- 2.0-3.5x speedup over baseline
- 70-95% memory reduction
- Custom kernel optimization opportunities

Learning Objectives:
- Understanding Triton GPU programming model
- Implementing custom kernels for ML operations
- Performance analysis of hand-optimized kernels
- Memory access pattern optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
import time
from typing import Optional, Tuple
from dataclasses import dataclass

# Triton kernel implementations
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMSNorm operation.
    Fuses variance computation and normalization in a single kernel.
    """
    row_idx = tl.program_id(0)

    # Compute variance in blocks
    variance = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x_vals = tl.load(x_ptr + row_idx * n_elements + offsets, mask=mask, other=0.0)
        variance += tl.sum(x_vals * x_vals, axis=0)

    variance = variance / n_elements
    inv_std = 1.0 / tl.sqrt(variance + eps)

    # Apply normalization in blocks
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x_vals = tl.load(x_ptr + row_idx * n_elements + offsets, mask=mask, other=0.0)
        weight_vals = tl.load(weight_ptr + offsets, mask=mask, other=1.0)

        normalized = x_vals * inv_std * weight_vals
        tl.store(output_ptr + row_idx * n_elements + offsets, normalized, mask=mask)


@triton.jit
def swiglu_kernel(
    x_ptr, gate_weight_ptr, up_weight_ptr, output_ptr,
    batch_size, seq_len, d_model, d_ff,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for fused SwiGLU operation.
    Combines gate and up projections with SiLU activation in single kernel.
    """
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    d_idx = tl.program_id(2)

    if batch_idx >= batch_size or seq_idx >= seq_len or d_idx >= d_ff:
        return

    # Load input
    input_offset = batch_idx * seq_len * d_model + seq_idx * d_model
    x_block = tl.load(x_ptr + input_offset + tl.arange(0, d_model))

    # Compute gate projection
    gate_sum = 0.0
    up_sum = 0.0

    for i in range(0, d_model, BLOCK_SIZE_D):
        x_vals = tl.load(x_ptr + input_offset + i + tl.arange(0, BLOCK_SIZE_D))
        gate_weights = tl.load(gate_weight_ptr + d_idx * d_model + i + tl.arange(0, BLOCK_SIZE_D))
        up_weights = tl.load(up_weight_ptr + d_idx * d_model + i + tl.arange(0, BLOCK_SIZE_D))

        gate_sum += tl.sum(x_vals * gate_weights)
        up_sum += tl.sum(x_vals * up_weights)

    # Apply SiLU activation to gate and multiply with up
    gate_activated = gate_sum / (1.0 + tl.exp(-gate_sum))
    result = gate_activated * up_sum

    # Store result
    output_offset = batch_idx * seq_len * d_ff + seq_idx * d_ff + d_idx
    tl.store(output_ptr + output_offset, result)


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Simplified Flash Attention kernel using Triton.
    Implements memory-efficient attention computation.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)

    # Calculate offsets
    head_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim

    # Load Q block
    q_start = q_block_idx * BLOCK_SIZE_Q
    q_offsets = q_start + tl.arange(0, BLOCK_SIZE_Q)
    q_mask = q_offsets < seq_len

    q_block = tl.zeros((BLOCK_SIZE_Q, head_dim), dtype=tl.float32)
    for d in range(head_dim):
        q_vals = tl.load(q_ptr + head_offset + q_offsets * head_dim + d, mask=q_mask, other=0.0)
        q_block = tl.where(tl.arange(0, head_dim)[None, :] == d, q_vals[:, None], q_block)

    # Initialize output accumulators
    output_acc = tl.zeros((BLOCK_SIZE_Q, head_dim), dtype=tl.float32)
    max_scores = tl.full((BLOCK_SIZE_Q,), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)

    # Process K,V blocks
    for k_block_start in range(0, seq_len, BLOCK_SIZE_K):
        k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < seq_len

        # Load K block
        k_block = tl.zeros((BLOCK_SIZE_K, head_dim), dtype=tl.float32)
        for d in range(head_dim):
            k_vals = tl.load(k_ptr + head_offset + k_offsets * head_dim + d, mask=k_mask, other=0.0)
            k_block = tl.where(tl.arange(0, head_dim)[None, :] == d, k_vals[:, None], k_block)

        # Compute attention scores
        scores = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        for d in range(head_dim):
            q_d = tl.load(q_ptr + head_offset + q_offsets * head_dim + d, mask=q_mask, other=0.0)
            k_d = tl.load(k_ptr + head_offset + k_offsets * head_dim + d, mask=k_mask, other=0.0)
            scores += q_d[:, None] * k_d[None, :] * scale

        # Apply causal mask
        causal_mask = q_offsets[:, None] >= k_offsets[None, :]
        scores = tl.where(causal_mask, scores, -float('inf'))

        # Update running statistics
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_scores, block_max)
        exp_scores = tl.exp(scores - new_max[:, None])

        # Update accumulated values
        decay = tl.exp(max_scores - new_max)
        sum_exp = sum_exp * decay + tl.sum(exp_scores, axis=1)
        max_scores = new_max

        # Load V block and accumulate output
        for d in range(head_dim):
            v_vals = tl.load(v_ptr + head_offset + k_offsets * head_dim + d, mask=k_mask, other=0.0)
            weighted_v = tl.sum(exp_scores * v_vals[None, :], axis=1)
            output_acc = output_acc + weighted_v[:, None] * (tl.arange(0, head_dim)[None, :] == d)

    # Final normalization and store
    output_normalized = output_acc / sum_exp[:, None]
    for d in range(head_dim):
        out_vals = output_normalized[:, d]
        tl.store(output_ptr + head_offset + q_offsets * head_dim + d, out_vals, mask=q_mask)


class TritonRMSNorm(nn.Module):
    """RMSNorm using custom Triton kernel."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.view(-1, dim)
        output = torch.empty_like(x_reshaped)

        grid = (x_reshaped.shape[0],)
        rmsnorm_kernel[grid](
            x_reshaped, self.weight, output,
            dim, self.eps, BLOCK_SIZE=256
        )

        return output.view(batch_size, seq_len, dim)


class TritonSwiGLU(nn.Module):
    """SwiGLU using custom Triton kernel."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        d_ff = self.gate_proj.out_features

        # Fused gate and up projections with SiLU
        intermediate = torch.empty(batch_size, seq_len, d_ff, device=x.device, dtype=x.dtype)

        grid = (batch_size, seq_len, d_ff)
        swiglu_kernel[grid](
            x, self.gate_proj.weight, self.up_proj.weight, intermediate,
            batch_size, seq_len, d_model, d_ff,
            BLOCK_SIZE_B=1, BLOCK_SIZE_S=1, BLOCK_SIZE_D=64
        )

        return self.down_proj(intermediate)


class TritonAttention(nn.Module):
    """Multi-head attention using custom Triton Flash Attention kernel."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Reshape for kernel
        q = q.transpose(1, 2).contiguous()  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Apply Flash Attention kernel
        output = torch.empty_like(q)

        grid = (batch_size, self.n_heads, triton.cdiv(seq_len, 64))
        flash_attention_kernel[grid](
            q, k, v, output,
            batch_size, self.n_heads, seq_len, self.head_dim,
            self.scale,
            BLOCK_SIZE_Q=64, BLOCK_SIZE_K=64
        )

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.o_proj(output)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (unchanged from previous versions)."""

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len: int):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)


class TransformerBlock(nn.Module):
    """Transformer block with Triton-optimized components."""

    def __init__(self, dim: int, n_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.attention = TritonAttention(dim, n_heads)
        self.feed_forward = TritonSwiGLU(dim, int(2.67 * dim))
        self.attention_norm = TritonRMSNorm(dim, norm_eps)
        self.ffn_norm = TritonRMSNorm(dim, norm_eps)

    def forward(self, x, rope_cos, rope_sin):
        # Attention with residual
        attn_input = self.attention_norm(x)
        attn_output = self.attention(attn_input)
        x = x + attn_output

        # Feed forward with residual
        ff_input = self.ffn_norm(x)
        ff_output = self.feed_forward(ff_input)
        x = x + ff_output

        return x


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    norm_eps: float = 1e-5
    max_seq_len: int = 2048


class TinyLlamaTriton(nn.Module):
    """Tiny LLaMA model with Triton kernel optimizations."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(config.dim, config.n_heads, config.norm_eps)
            for _ in range(config.n_layers)
        ])
        self.norm = TritonRMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.rope = RotaryPositionEmbedding(config.dim // config.n_heads, config.max_seq_len)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.embed_tokens(input_ids)

        # Rotary position embeddings
        cos, sin = self.rope(x, seq_len)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, cos, sin)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


def benchmark_triton_model():
    """Benchmark the Triton-optimized model."""
    print("=== Triton Kernel Model Benchmark ===")

    # Model configuration
    config = ModelConfig(
        vocab_size=32000,
        dim=2048,
        n_layers=16,
        n_heads=32,
        norm_eps=1e-5,
        max_seq_len=1024
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyLlamaTriton(config).to(device)

    # Prepare input
    batch_size = 4
    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"Input shape: {input_ids.shape}")

    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)

    torch.cuda.synchronize()

    # Benchmark forward pass
    num_runs = 50
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(input_ids)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    throughput = batch_size * seq_len / avg_time

    print(f"Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.0f} tokens/second")
    print(f"Memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Calculate FLOPS
    total_params = sum(p.numel() for p in model.parameters())
    approx_flops = 2 * total_params * batch_size * seq_len  # Rough estimate
    flops_per_sec = approx_flops / avg_time

    print(f"Estimated FLOPS/second: {flops_per_sec / 1e12:.2f} TFLOPS")

    return avg_time, throughput, torch.cuda.max_memory_allocated()


if __name__ == "__main__":
    # Set deterministic behavior
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run benchmark
    benchmark_triton_model()