#!/usr/bin/env python3
"""
Tiny OpenFold V1: PyTorch Baseline with Comprehensive Profiling Integration

Educational implementation of AlphaFold 2's Evoformer architecture for protein structure prediction.
This version integrates PyTorch Profiler and comprehensive performance analysis capabilities
while maintaining deterministic execution.

Features:
- Evoformer blocks with MSA and pair representations
- Triangle multiplicative updates for geometric reasoning
- MSA row/column attention mechanisms
- PyTorch Profiler integration with GPU/CPU timeline analysis
- Memory profiling and bandwidth analysis
- Operator-level performance characterization
- Comprehensive performance reporting

Usage:
    # Basic training
    python tiny_openfold_v1.py --batch-size 4 --seq-len 64

    # With PyTorch profiler
    python tiny_openfold_v1.py --enable-pytorch-profiler --profile-dir ./profiles

    # With memory profiling
    python tiny_openfold_v1.py --enable-pytorch-profiler --profile-memory

    # Complete profiling suite
    python tiny_openfold_v1.py --enable-all-profiling --profile-dir ./complete_analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
import time
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Optional imports with graceful fallbacks
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False
    class nvtx:
        @staticmethod
        def range(name):
            from contextlib import nullcontext
            return nullcontext()


@dataclass
class TinyOpenFoldConfig:
    """Configuration for Tiny OpenFold model - optimized for profiling."""
    vocab_size: int = 21                # 20 amino acids + unknown
    msa_dim: int = 64                   # MSA representation dimension
    pair_dim: int = 128                 # Pair representation dimension
    n_evoformer_blocks: int = 4         # Number of Evoformer blocks
    n_heads_msa: int = 4                # Number of MSA attention heads
    n_heads_pair: int = 4               # Number of pair attention heads
    msa_intermediate_dim: int = 256     # MSA transition intermediate dimension
    pair_intermediate_dim: int = 512    # Pair transition intermediate dimension
    outer_product_dim: int = 32         # Outer product mean dimension
    max_seq_len: int = 64               # Maximum sequence length
    n_seqs: int = 16                    # Number of MSA sequences
    pair_input_dim: int = 65            # Pair input features (distance bins, etc.)
    dropout: float = 0.0                # Dropout rate (0 for profiling)
    norm_eps: float = 1e-5              # Layer norm epsilon

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ProfilerConfig:
    """Configuration for profiling options."""
    enable_pytorch_profiler: bool = False
    enable_memory_profiling: bool = False
    profile_operators: bool = False
    profile_dir: str = "./pytorch_profiles"
    sort_by: str = "cuda_time_total"
    warmup_steps: int = 3
    profile_steps: int = 5
    export_chrome_trace: bool = True
    export_stacks: bool = False


class PerformanceMonitor:
    """Comprehensive performance monitoring and analysis."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'training_speed': [],
            'memory_usage': [],
            'loss_values': [],
            'batch_times': [],
            'forward_times': [],
            'backward_times': [],
            'optimizer_times': []
        }
        self.start_time = None
        self.total_samples = 0

    def start_timing(self):
        """Start timing measurement."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()

    def end_timing(self) -> float:
        """End timing measurement and return elapsed time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed

    def record_batch_metrics(self, batch_size: int, loss: float, timings: Dict[str, float]):
        """Record metrics for a training batch."""
        self.total_samples += batch_size
        self.metrics['loss_values'].append(loss)
        self.metrics['batch_times'].append(timings.get('total', 0))
        self.metrics['forward_times'].append(timings.get('forward', 0))
        self.metrics['backward_times'].append(timings.get('backward', 0))
        self.metrics['optimizer_times'].append(timings.get('optimizer', 0))

        # Memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024**2)
            self.metrics['memory_usage'].append(memory_mb)

        # Training speed (samples per second)
        if timings.get('total', 0) > 0:
            speed = batch_size / timings['total']
            self.metrics['training_speed'].append(speed)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics['batch_times']:
            return {}

        summary = {
            'total_samples': self.total_samples,
            'avg_training_speed': np.mean(self.metrics['training_speed']) if self.metrics['training_speed'] else 0,
            'avg_loss': np.mean(self.metrics['loss_values']),
            'avg_batch_time': np.mean(self.metrics['batch_times']),
            'avg_forward_time': np.mean(self.metrics['forward_times']),
            'avg_backward_time': np.mean(self.metrics['backward_times']),
            'avg_optimizer_time': np.mean(self.metrics['optimizer_times']),
        }

        if self.metrics['memory_usage']:
            summary.update({
                'peak_memory_mb': max(self.metrics['memory_usage']),
                'avg_memory_mb': np.mean(self.metrics['memory_usage'])
            })

        return summary


def get_available_devices() -> Tuple[list, bool]:
    """
    Detect available GPUs respecting ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES.
    
    Returns:
        (device_ids, multi_gpu): List of available device IDs and whether multi-GPU is enabled
    """
    if not torch.cuda.is_available():
        return [], False
    
    # Check environment variables (priority: ROCR > HIP > CUDA)
    rocr_devices = os.environ.get('ROCR_VISIBLE_DEVICES')
    hip_devices = os.environ.get('HIP_VISIBLE_DEVICES')
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    env_devices = rocr_devices or hip_devices or cuda_devices
    
    if env_devices:
        # Parse comma-separated device IDs
        try:
            device_ids = [int(d.strip()) for d in env_devices.split(',') if d.strip().isdigit()]
            if not device_ids:
                # If parsing failed, use all available
                device_ids = list(range(torch.cuda.device_count()))
        except ValueError:
            device_ids = list(range(torch.cuda.device_count()))
    else:
        # Use all available devices
        device_ids = list(range(torch.cuda.device_count()))
    
    # Filter device_ids to only those actually available
    device_ids = [d for d in device_ids if d < torch.cuda.device_count()]
    
    multi_gpu = len(device_ids) > 1
    return device_ids, multi_gpu


def setup_deterministic_environment():
    """Configure PyTorch for deterministic execution."""
    seed = 42

    # Python random
    import random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA/ROCm
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Deterministic execution environment configured")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class MSARowAttentionWithPairBias(nn.Module):
    """MSA row-wise attention biased by pair representation."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()
        self.msa_dim = config.msa_dim
        self.n_heads = config.n_heads_msa
        self.head_dim = config.msa_dim // config.n_heads_msa
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections for MSA
        self.q_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
        self.k_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
        self.v_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
        self.o_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)

        # Pair bias projection
        self.pair_bias_proj = nn.Linear(config.pair_dim, config.n_heads_msa, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa: (batch, n_seqs, seq_len, msa_dim)
            pair: (batch, seq_len, seq_len, pair_dim)
        Returns:
            (batch, n_seqs, seq_len, msa_dim)
        """
        with record_function("msa_row_attention"):
            batch_size, n_seqs, seq_len, _ = msa.shape

            # Project to Q, K, V
            with record_function("msa_qkv_projection"):
                q = self.q_proj(msa).view(batch_size, n_seqs, seq_len, self.n_heads, self.head_dim)
                k = self.k_proj(msa).view(batch_size, n_seqs, seq_len, self.n_heads, self.head_dim)
                v = self.v_proj(msa).view(batch_size, n_seqs, seq_len, self.n_heads, self.head_dim)

                # Transpose for attention: (batch, n_seqs, n_heads, seq_len, head_dim)
                q = q.transpose(2, 3)
                k = k.transpose(2, 3)
                v = v.transpose(2, 3)

            # Compute attention scores
            with record_function("msa_attention_scores"):
                # (batch, n_seqs, n_heads, seq_len, seq_len)
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                # Add pair bias: (batch, seq_len, seq_len, pair_dim) -> (batch, n_heads, seq_len, seq_len)
                pair_bias = self.pair_bias_proj(pair).permute(0, 3, 1, 2)
                scores = scores + pair_bias.unsqueeze(1)  # Broadcast across n_seqs

            # Apply softmax and dropout
            with record_function("msa_attention_softmax"):
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            with record_function("msa_attention_output"):
                attn_output = torch.matmul(attn_weights, v)
                # (batch, n_seqs, n_heads, seq_len, head_dim) -> (batch, n_seqs, seq_len, msa_dim)
                attn_output = attn_output.transpose(2, 3).contiguous().view(batch_size, n_seqs, seq_len, self.msa_dim)
                output = self.o_proj(attn_output)

            return output


class MSAColumnAttention(nn.Module):
    """MSA column-wise attention (across sequences)."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()
        self.msa_dim = config.msa_dim
        self.n_heads = config.n_heads_msa
        self.head_dim = config.msa_dim // config.n_heads_msa
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
        self.k_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
        self.v_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)
        self.o_proj = nn.Linear(config.msa_dim, config.msa_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa: (batch, n_seqs, seq_len, msa_dim)
        Returns:
            (batch, n_seqs, seq_len, msa_dim)
        """
        with record_function("msa_column_attention"):
            batch_size, n_seqs, seq_len, _ = msa.shape

            # Transpose to put seq_len first for column-wise attention
            # (batch, seq_len, n_seqs, msa_dim)
            msa_t = msa.transpose(1, 2)

            # Project to Q, K, V
            with record_function("msa_col_qkv_projection"):
                q = self.q_proj(msa_t).view(batch_size, seq_len, n_seqs, self.n_heads, self.head_dim)
                k = self.k_proj(msa_t).view(batch_size, seq_len, n_seqs, self.n_heads, self.head_dim)
                v = self.v_proj(msa_t).view(batch_size, seq_len, n_seqs, self.n_heads, self.head_dim)

                # Transpose for attention: (batch, seq_len, n_heads, n_seqs, head_dim)
                q = q.transpose(2, 3)
                k = k.transpose(2, 3)
                v = v.transpose(2, 3)

            # Compute attention scores
            with record_function("msa_col_attention_scores"):
                # (batch, seq_len, n_heads, n_seqs, n_seqs)
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply softmax and dropout
            with record_function("msa_col_attention_softmax"):
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            with record_function("msa_col_attention_output"):
                attn_output = torch.matmul(attn_weights, v)
                # (batch, seq_len, n_heads, n_seqs, head_dim) -> (batch, seq_len, n_seqs, msa_dim)
                attn_output = attn_output.transpose(2, 3).contiguous().view(batch_size, seq_len, n_seqs, self.msa_dim)
                output = self.o_proj(attn_output)

            # Transpose back to (batch, n_seqs, seq_len, msa_dim)
            return output.transpose(1, 2)


class MSATransition(nn.Module):
    """Point-wise feed-forward network for MSA."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.msa_dim, config.msa_intermediate_dim, bias=False)
        self.linear2 = nn.Linear(config.msa_intermediate_dim, config.msa_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        with record_function("msa_transition"):
            x = self.linear1(msa)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return self.dropout(x)


class OuterProductMean(nn.Module):
    """Outer product mean: projects MSA to pair representation."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()
        self.msa_to_outer = nn.Linear(config.msa_dim, config.outer_product_dim, bias=False)
        self.outer_to_pair = nn.Linear(config.outer_product_dim ** 2, config.pair_dim, bias=False)
        self.layer_norm = nn.LayerNorm(config.msa_dim, eps=config.norm_eps)

    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa: (batch, n_seqs, seq_len, msa_dim)
        Returns:
            pair_update: (batch, seq_len, seq_len, pair_dim)
        """
        with record_function("outer_product_mean"):
            batch_size, n_seqs, seq_len, _ = msa.shape

            # Normalize and project
            msa_norm = self.layer_norm(msa)
            outer_features = self.msa_to_outer(msa_norm)  # (batch, n_seqs, seq_len, outer_dim)

            # Compute outer product between all position pairs, mean over sequences
            with record_function("outer_product_computation"):
                # Einstein summation: for positions i,j compute mean_n(feat[n,i] ⊗ feat[n,j])
                # bnid: batch, n_seqs, position_i, outer_dim
                # bnje: batch, n_seqs, position_j, outer_dim
                # bijde: batch, position_i, position_j, outer_dim, outer_dim
                outer = torch.einsum('bnid,bnje->bijde', outer_features, outer_features) / n_seqs
                # outer: (batch, seq_len, seq_len, outer_dim, outer_dim)

                # Flatten last two dimensions
                outer_flat = outer.flatten(-2, -1)  # (batch, seq_len, seq_len, outer_dim²)

            # Project to pair dimension
            pair_update = self.outer_to_pair(outer_flat)

            return pair_update


class TriangleMultiplication(nn.Module):
    """Triangle multiplicative update (outgoing or incoming)."""

    def __init__(self, config: TinyOpenFoldConfig, outgoing: bool = True):
        super().__init__()
        self.outgoing = outgoing

        # Gated projections
        self.left_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.right_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.left_gate = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.right_gate = nn.Linear(config.pair_dim, config.pair_dim, bias=False)

        # Output projection and gate
        self.output_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.output_gate = nn.Linear(config.pair_dim, config.pair_dim, bias=False)

        self.layer_norm = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair: (batch, seq_len, seq_len, pair_dim)
        Returns:
            (batch, seq_len, seq_len, pair_dim)
        """
        name = "triangle_mult_outgoing" if self.outgoing else "triangle_mult_incoming"
        with record_function(name):
            pair_norm = self.layer_norm(pair)

            # Compute left and right projections with gates
            left = self.left_proj(pair_norm) * torch.sigmoid(self.left_gate(pair_norm))
            right = self.right_proj(pair_norm) * torch.sigmoid(self.right_gate(pair_norm))

            # Triangle multiplication
            with record_function(f"{name}_matmul"):
                if self.outgoing:
                    # Sum over k: z_ij += left_ik * right_jk
                    update = torch.einsum('bikc,bjkc->bijc', left, right)
                else:
                    # Sum over k: z_ij += left_ki * right_kj
                    update = torch.einsum('bkic,bkjc->bijc', left, right)

            # Output projection with gate
            gate = torch.sigmoid(self.output_gate(pair_norm))
            output = self.output_proj(update) * gate

            return output


class TriangleAttention(nn.Module):
    """Triangle self-attention (starting or ending node)."""

    def __init__(self, config: TinyOpenFoldConfig, starting: bool = True):
        super().__init__()
        self.starting = starting
        self.n_heads = config.n_heads_pair
        self.head_dim = config.pair_dim // config.n_heads_pair
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.k_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.v_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False)
        self.o_proj = nn.Linear(config.pair_dim, config.pair_dim, bias=False)

        self.layer_norm = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair: (batch, seq_len, seq_len, pair_dim)
        Returns:
            (batch, seq_len, seq_len, pair_dim)
        """
        name = "triangle_attn_starting" if self.starting else "triangle_attn_ending"
        with record_function(name):
            batch_size, seq_len, _, pair_dim = pair.shape
            pair_norm = self.layer_norm(pair)

            if self.starting:
                # Attention over edges starting from a node: fix i, attend over j
                q = self.q_proj(pair_norm).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
                k = self.k_proj(pair_norm).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
                v = self.v_proj(pair_norm).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)

                # (batch, seq_len, n_heads, seq_len, head_dim)
                q = q.transpose(2, 3)
                k = k.transpose(2, 3)
                v = v.transpose(2, 3)

                # Attention: (batch, seq_len, n_heads, seq_len, seq_len)
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(scores, dim=-1)

                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(2, 3).contiguous().view(batch_size, seq_len, seq_len, pair_dim)
            else:
                # Attention over edges ending at a node: fix j, attend over i
                # Transpose to make j the "batch" dimension
                pair_t = pair_norm.transpose(1, 2)  # (batch, seq_len, seq_len, pair_dim)

                q = self.q_proj(pair_t).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
                k = self.k_proj(pair_t).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)
                v = self.v_proj(pair_t).view(batch_size, seq_len, seq_len, self.n_heads, self.head_dim)

                q = q.transpose(2, 3)
                k = k.transpose(2, 3)
                v = v.transpose(2, 3)

                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(scores, dim=-1)

                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(2, 3).contiguous().view(batch_size, seq_len, seq_len, pair_dim)

                # Transpose back
                attn_output = attn_output.transpose(1, 2)

            output = self.o_proj(attn_output)
            return output


class PairTransition(nn.Module):
    """Point-wise feed-forward network for pair representation."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.pair_dim, config.pair_intermediate_dim, bias=False)
        self.linear2 = nn.Linear(config.pair_intermediate_dim, config.pair_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        with record_function("pair_transition"):
            x = self.linear1(pair)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            return self.dropout(x)


class EvoformerBlock(nn.Module):
    """Single Evoformer block with MSA and pair representation updates."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()

        # MSA operations
        self.msa_row_attention = MSARowAttentionWithPairBias(config)
        self.msa_column_attention = MSAColumnAttention(config)
        self.msa_transition = MSATransition(config)

        # MSA layer norms
        self.msa_norm_row = nn.LayerNorm(config.msa_dim, eps=config.norm_eps)
        self.msa_norm_col = nn.LayerNorm(config.msa_dim, eps=config.norm_eps)
        self.msa_norm_trans = nn.LayerNorm(config.msa_dim, eps=config.norm_eps)

        # Pair operations
        self.outer_product_mean = OuterProductMean(config)
        self.triangle_mult_outgoing = TriangleMultiplication(config, outgoing=True)
        self.triangle_mult_incoming = TriangleMultiplication(config, outgoing=False)
        self.triangle_attn_starting = TriangleAttention(config, starting=True)
        self.triangle_attn_ending = TriangleAttention(config, starting=False)
        self.pair_transition = PairTransition(config)

        # Pair layer norms
        self.pair_norm_outer = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)
        self.pair_norm_tri_out = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)
        self.pair_norm_tri_in = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)
        self.pair_norm_attn_start = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)
        self.pair_norm_attn_end = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)
        self.pair_norm_trans = nn.LayerNorm(config.pair_dim, eps=config.norm_eps)

    def forward(self, msa: torch.Tensor, pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            msa: (batch, n_seqs, seq_len, msa_dim)
            pair: (batch, seq_len, seq_len, pair_dim)
        Returns:
            msa, pair (same shapes as input)
        """
        with record_function("evoformer_block"):
            # MSA updates
            with record_function("evoformer_msa_updates"):
                msa = msa + self.msa_row_attention(self.msa_norm_row(msa), pair)
                msa = msa + self.msa_column_attention(self.msa_norm_col(msa))
                msa = msa + self.msa_transition(self.msa_norm_trans(msa))

            # Pair updates
            with record_function("evoformer_pair_updates"):
                pair = pair + self.outer_product_mean(msa)
                pair = pair + self.triangle_mult_outgoing(self.pair_norm_tri_out(pair))
                pair = pair + self.triangle_mult_incoming(self.pair_norm_tri_in(pair))
                pair = pair + self.triangle_attn_starting(self.pair_norm_attn_start(pair))
                pair = pair + self.triangle_attn_ending(self.pair_norm_attn_end(pair))
                pair = pair + self.pair_transition(self.pair_norm_trans(pair))

            return msa, pair


class SimplifiedStructureModule(nn.Module):
    """Simplified structure module: predicts distances from pair representation."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()
        # Predict pairwise distances
        self.distance_pred = nn.Linear(config.pair_dim, 1, bias=False)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair: (batch, seq_len, seq_len, pair_dim)
        Returns:
            distances: (batch, seq_len, seq_len, 1)
        """
        with record_function("structure_module"):
            distances = self.distance_pred(pair)
            # Apply sigmoid to constrain to reasonable range
            distances = torch.sigmoid(distances) * 20.0  # Scale to ~20 Angstroms
            return distances


class TinyOpenFold(nn.Module):
    """Tiny OpenFold model for protein structure prediction."""

    def __init__(self, config: TinyOpenFoldConfig):
        super().__init__()
        self.config = config

        # Input embeddings
        self.msa_embedding = nn.Embedding(config.vocab_size, config.msa_dim)
        self.pair_embedding = nn.Linear(config.pair_input_dim, config.pair_dim, bias=False)

        # Evoformer blocks
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(config) for _ in range(config.n_evoformer_blocks)
        ])

        # Structure module
        self.structure_module = SimplifiedStructureModule(config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, msa_tokens: torch.Tensor, pair_features: torch.Tensor,
                target_distances: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            msa_tokens: (batch, n_seqs, seq_len) - amino acid tokens
            pair_features: (batch, seq_len, seq_len, pair_input_dim) - pairwise features
            target_distances: (batch, seq_len, seq_len, 1) - ground truth distances (optional)
        Returns:
            dict with 'distances' and optionally 'loss'
        """
        with record_function("model_forward"):
            # Embed inputs
            with record_function("input_embedding"):
                msa = self.msa_embedding(msa_tokens)  # (batch, n_seqs, seq_len, msa_dim)
                pair = self.pair_embedding(pair_features)  # (batch, seq_len, seq_len, pair_dim)

            # Pass through Evoformer blocks
            with record_function("evoformer_layers"):
                for i, block in enumerate(self.evoformer_blocks):
                    with record_function(f"evoformer_{i}"):
                        msa, pair = block(msa, pair)

            # Predict structure
            with record_function("structure_prediction"):
                predicted_distances = self.structure_module(pair)

            # Calculate loss if targets provided
            loss = None
            if target_distances is not None:
                with record_function("loss_calculation"):
                    # MSE loss on distances
                    loss = F.mse_loss(predicted_distances, target_distances)

            return {
                'distances': predicted_distances,
                'loss': loss,
                'pair_repr': pair,
                'msa_repr': msa
            }


class ProteinDataset:
    """Synthetic protein dataset for training demonstration."""

    def __init__(self, config: TinyOpenFoldConfig, num_samples: int = 1000):
        self.config = config
        self.num_samples = num_samples

        # Generate synthetic data (deterministic)
        np.random.seed(42)

        # Random MSA sequences
        self.msa_data = np.random.randint(
            0, config.vocab_size,
            size=(num_samples, config.n_seqs, config.max_seq_len),
            dtype=np.int64
        )

        # Random pair features (e.g., distance bins)
        self.pair_data = np.random.randn(
            num_samples, config.max_seq_len, config.max_seq_len, config.pair_input_dim
        ).astype(np.float32)

        # Random target distances (simulate true structure)
        self.distance_data = np.random.rand(
            num_samples, config.max_seq_len, config.max_seq_len, 1
        ).astype(np.float32) * 20.0  # 0-20 Angstroms

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        indices = np.random.choice(self.num_samples, batch_size, replace=False)

        msa_tokens = torch.from_numpy(self.msa_data[indices])
        pair_features = torch.from_numpy(self.pair_data[indices])
        target_distances = torch.from_numpy(self.distance_data[indices])

        return msa_tokens, pair_features, target_distances


def setup_pytorch_profiler(profiler_config: ProfilerConfig) -> Optional[profile]:
    """Setup PyTorch profiler with comprehensive configuration."""
    if not profiler_config.enable_pytorch_profiler:
        return None

    # Ensure profile directory exists
    Path(profiler_config.profile_dir).mkdir(parents=True, exist_ok=True)

    # Profiler activities
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Profiler configuration
    profiler = profile(
        activities=activities,
        record_shapes=True,
        profile_memory=profiler_config.enable_memory_profiling,
        with_stack=profiler_config.export_stacks,
        with_flops=True,
        with_modules=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True
        ),
        schedule=torch.profiler.schedule(
            wait=profiler_config.warmup_steps,
            warmup=1,
            active=profiler_config.profile_steps,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_config.profile_dir)
    )

    return profiler


def train_tiny_openfold(
    config: TinyOpenFoldConfig,
    profiler_config: ProfilerConfig,
    num_steps: int = 50,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    use_amp: bool = False,
    device_id: Optional[int] = None,
    use_data_parallel: bool = True
):
    """Train the Tiny OpenFold model with comprehensive profiling (single or multi-GPU)."""

    # Setup environment
    setup_deterministic_environment()
    
    # Detect available devices
    available_devices, multi_gpu_available = get_available_devices()
    
    # Device selection logic
    if device_id is not None:
        # Single device mode (explicit selection overrides everything)
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Device {device_id} not available. Only {torch.cuda.device_count()} GPU(s) found.")
        device = torch.device(f"cuda:{device_id}")
        use_multi_gpu = False
        print(f"\n   Single GPU mode: Using cuda:{device_id} (explicit)")
    elif multi_gpu_available and use_data_parallel and len(available_devices) > 1:
        # Multi-GPU mode
        device = torch.device(f"cuda:{available_devices[0]}")  # Primary device
        use_multi_gpu = True
        
        # Show environment variable that was used
        env_var = "ROCR_VISIBLE_DEVICES" if os.environ.get('ROCR_VISIBLE_DEVICES') else \
                  "HIP_VISIBLE_DEVICES" if os.environ.get('HIP_VISIBLE_DEVICES') else \
                  "CUDA_VISIBLE_DEVICES" if os.environ.get('CUDA_VISIBLE_DEVICES') else \
                  "all available"
        
        print(f"\n   Multi-GPU mode: Using {len(available_devices)} GPUs")
        print(f"   Device IDs: {available_devices} (from {env_var})")
        print(f"   Primary device: cuda:{available_devices[0]}")
        print(f"   Effective batch size: {batch_size} total (split across GPUs)")
    else:
        # Default single GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_multi_gpu = False
        print(f"\n   Single GPU mode: Using default device ({device})")
    
    # Ensure profile directory exists
    if profiler_config.profile_dir:
        Path(profiler_config.profile_dir).mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = TinyOpenFold(config)
    
    # Wrap with DataParallel if multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(model, device_ids=available_devices)
        print(f"   Model wrapped with DataParallel")
    
    model = model.to(device)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Configuration:")
    print(f"   MSA dimension: {config.msa_dim}")
    print(f"   Pair dimension: {config.pair_dim}")
    print(f"   Evoformer blocks: {config.n_evoformer_blocks}")
    print(f"   MSA sequences: {config.n_seqs}")
    print(f"   Sequence length: {config.max_seq_len}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1e6:.1f} MB (FP32)")
    
    if isinstance(model, nn.DataParallel):
        print(f"   Multi-GPU: {len(model.device_ids)} GPUs")
        print(f"   Device IDs: {model.device_ids}")
        print(f"   Primary device: {device}")
    else:
        print(f"   Device: {device}")

    # Create dataset
    dataset = ProteinDataset(config)

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Setup mixed precision
    scaler = GradScaler() if use_amp else None

    # Setup profiler
    pytorch_profiler = setup_pytorch_profiler(profiler_config)

    # Performance monitor
    monitor = PerformanceMonitor()

    print(f"\nTraining Configuration:")
    print(f"   Training steps: {num_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Mixed precision: {use_amp}")
    print(f"   Device: {device}")
    print(f"   PyTorch Profiler: {profiler_config.enable_pytorch_profiler}")
    print(f"   Memory Profiling: {profiler_config.enable_memory_profiling}")

    # Training loop
    model.train()

    # Warmup steps
    warmup_steps = 5
    print(f"\nRunning {warmup_steps} warmup steps...")

    for step in range(warmup_steps):
        msa_tokens, pair_features, target_distances = dataset.get_batch(batch_size)
        msa_tokens = msa_tokens.to(device)
        pair_features = pair_features.to(device)
        target_distances = target_distances.to(device)

        if use_amp:
            with autocast():
                outputs = model(msa_tokens, pair_features, target_distances)
                loss = outputs['loss'].mean()  # Average loss across GPUs for DataParallel
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(msa_tokens, pair_features, target_distances)
            loss = outputs['loss'].mean()  # Average loss across GPUs for DataParallel
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

    print(f"Warmup complete. Starting measured training loop...")
    print("=" * 70)

    for step in range(num_steps):
        # Start batch timing
        batch_timings = {}
        monitor.start_timing()

        # Get batch
        with nvtx.range("data_loading"):
            msa_tokens, pair_features, target_distances = dataset.get_batch(batch_size)
            msa_tokens = msa_tokens.to(device)
            pair_features = pair_features.to(device)
            target_distances = target_distances.to(device)

        # Forward pass timing
        monitor.start_timing()
        with nvtx.range("forward_pass"):
            if use_amp:
                with autocast():
                    outputs = model(msa_tokens, pair_features, target_distances)
                    loss = outputs['loss'].mean()  # Average loss across GPUs for DataParallel
            else:
                outputs = model(msa_tokens, pair_features, target_distances)
                loss = outputs['loss'].mean()  # Average loss across GPUs for DataParallel
        batch_timings['forward'] = monitor.end_timing()

        # Backward pass timing
        monitor.start_timing()
        with nvtx.range("backward_pass"):
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        batch_timings['backward'] = monitor.end_timing()

        # Optimizer step timing
        monitor.start_timing()
        with nvtx.range("optimizer_step"):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        batch_timings['optimizer'] = monitor.end_timing()

        # Total batch time
        batch_timings['total'] = sum(batch_timings.values())

        # Record metrics
        monitor.record_batch_metrics(batch_size, loss.item(), batch_timings)

        # PyTorch profiler step
        if pytorch_profiler:
            pytorch_profiler.step()

        # Progress logging
        if step % 10 == 0:
            speed = batch_size / batch_timings['total'] if batch_timings['total'] > 0 else 0
            memory_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

            print(f"Step {step:3d}/{num_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Speed: {speed:5.1f} samples/sec | "
                  f"Memory: {memory_mb:6.1f} MB | "
                  f"Time: {batch_timings['total']*1000:5.1f}ms")

    print("=" * 70)

    # Performance summary
    summary = monitor.get_summary()
    avg_speed = summary.get('avg_training_speed', 0)

    print(f"\nPerformance Summary:")
    print(f"   Total samples processed: {summary.get('total_samples', 0):,}")
    print(f"   Average training speed: {avg_speed:.1f} samples/sec")
    print(f"   Average batch time: {summary.get('avg_batch_time', 0)*1000:.1f} ms")
    print(f"   Average forward time: {summary.get('avg_forward_time', 0)*1000:.1f} ms")
    print(f"   Average backward time: {summary.get('avg_backward_time', 0)*1000:.1f} ms")
    print(f"   Average optimizer time: {summary.get('avg_optimizer_time', 0)*1000:.1f} ms")
    print(f"   Final loss: {summary.get('avg_loss', 0):.4f}")

    if 'peak_memory_mb' in summary:
        print(f"   Peak memory usage: {summary['peak_memory_mb']:.1f} MB")

    # Save performance data
    if profiler_config.profile_dir:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        profile_data = {
            'version': 'v1_baseline',
            'timestamp': timestamp_str,
            'config': config.to_dict(),
            'profiler_config': asdict(profiler_config),
            'performance_summary': summary,
            'training_params': {
                'num_steps': num_steps,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'use_amp': use_amp
            },
            'system_info': {
                'device': str(device),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'pytorch_version': torch.__version__,
                'rocm_version': os.environ.get('ROCM_VERSION', 'N/A'),
                'timestamp_iso': datetime.now().isoformat()
            }
        }

        profile_path = Path(profiler_config.profile_dir) / "performance_summary.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)

        print(f"\nPerformance data saved to: {profile_path}")

    return model, monitor


def main():
    """Main entry point for Version 1 training."""
    parser = argparse.ArgumentParser(description='Tiny OpenFold V1: PyTorch Baseline with Profiling')

    # Model configuration
    parser.add_argument('--msa-dim', type=int, default=64, help='MSA dimension')
    parser.add_argument('--pair-dim', type=int, default=128, help='Pair dimension')
    parser.add_argument('--num-blocks', type=int, default=4, help='Number of Evoformer blocks')
    parser.add_argument('--num-seqs', type=int, default=16, help='Number of MSA sequences')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')

    # Training configuration
    parser.add_argument('--num-steps', type=int, default=50, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (total across all GPUs)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--device', type=int, default=None, help='Single GPU device index (disables multi-GPU)')
    parser.add_argument('--no-data-parallel', action='store_true', help='Disable DataParallel even if multiple GPUs available')

    # Profiling configuration
    parser.add_argument('--enable-pytorch-profiler', action='store_true', help='Enable PyTorch profiler')
    parser.add_argument('--enable-memory-profiling', action='store_true', help='Enable memory profiling')
    parser.add_argument('--enable-all-profiling', action='store_true', help='Enable all profiling features')
    parser.add_argument('--profile-operators', action='store_true', help='Profile individual operators')
    parser.add_argument('--profile-dir', type=str, default='./pytorch_profiles', help='Profiling output directory')
    parser.add_argument('--sort-by', type=str, default='cuda_time_total', help='Sort profiling results by metric')
    parser.add_argument('--warmup-steps', type=int, default=3, help='Profiler warmup steps')
    parser.add_argument('--profile-steps', type=int, default=5, help='Number of profiling steps')

    # Validation and debugging
    parser.add_argument('--validate-setup', action='store_true', help='Run validation checks')

    args = parser.parse_args()

    # Print banner
    print("=" * 80)
    print("TINY OPENFOLD - VERSION 1: PYTORCH BASELINE")
    print("     Educational AlphaFold 2 / Evoformer Implementation")
    print("=" * 80)

    # Configure model
    config = TinyOpenFoldConfig(
        msa_dim=args.msa_dim,
        pair_dim=args.pair_dim,
        n_evoformer_blocks=args.num_blocks,
        n_seqs=args.num_seqs,
        max_seq_len=args.seq_len,
        msa_intermediate_dim=args.msa_dim * 4,
        pair_intermediate_dim=args.pair_dim * 4
    )

    # Configure profiler
    profiler_config = ProfilerConfig(
        enable_pytorch_profiler=args.enable_pytorch_profiler or args.enable_all_profiling,
        enable_memory_profiling=args.enable_memory_profiling or args.enable_all_profiling,
        profile_operators=args.profile_operators,
        profile_dir=args.profile_dir,
        sort_by=args.sort_by,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps
    )

    # Validation mode
    if args.validate_setup:
        print("Running validation checks...")
        try:
            # Quick validation run
            model, monitor = train_tiny_openfold(
                config=config,
                profiler_config=profiler_config,
                num_steps=3,
                batch_size=2,
                device_id=args.device,
                use_data_parallel=not args.no_data_parallel
            )
            print("Validation successful! Environment ready.")
            return
        except Exception as e:
            print(f"Validation failed: {e}")
            return

    # Run training with profiling
    try:
        model, monitor = train_tiny_openfold(
            config=config,
            profiler_config=profiler_config,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_amp=args.use_amp,
            device_id=args.device,
            use_data_parallel=not args.no_data_parallel
        )

        print(f"\nTraining completed successfully!")

        if profiler_config.enable_pytorch_profiler:
            print(f"PyTorch profiling data saved to: {args.profile_dir}")
            print(f"   Launch TensorBoard: tensorboard --logdir {args.profile_dir}")

        print(f"\nNext Steps:")
        print(f"   1. Analyze profiling results to identify bottlenecks")
        print(f"   2. Review performance metrics and optimization opportunities")
        print(f"   3. Experiment with different configurations")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

