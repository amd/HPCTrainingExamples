#!/usr/bin/env python3
"""
Tiny LLaMA V1: PyTorch Baseline with Comprehensive Profiling Integration

Enhanced version of the baseline Tiny LLaMA implementation for the Castille AI Workshop.
This version integrates PyTorch Profiler, DeepSpeed FLOPS profiler, and comprehensive
performance analysis capabilities while maintaining deterministic execution.

Features:
- PyTorch Profiler integration with GPU/CPU timeline analysis
- DeepSpeed FLOPS profiler for computational efficiency metrics
- Memory profiling and bandwidth analysis
- Operator-level performance characterization
- Bottleneck identification and analysis
- Comprehensive performance reporting

Usage:
    # Basic training
    python tiny_llama_v1.py --batch-size 8 --seq-len 128

    # With PyTorch profiler
    python tiny_llama_v1.py --enable-pytorch-profiler --profile-dir ./profiles

    # With memory profiling
    python tiny_llama_v1.py --enable-pytorch-profiler --profile-memory

    # Complete profiling suite
    python tiny_llama_v1.py --enable-all-profiling --profile-dir ./complete_analysis
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

try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("Warning: DeepSpeed not available. FLOPS profiling disabled.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class TinyLlamaConfig:
    """Configuration for Tiny LLaMA model - optimized for profiling."""
    vocab_size: int = 1000          # Smaller for workshop
    hidden_dim: int = 256           # Model dimension
    n_layers: int = 4              # Number of transformer layers
    n_heads: int = 8               # Number of attention heads
    n_kv_heads: int = 4            # Number of key-value heads (for GQA)
    intermediate_dim: int = 512     # FFN intermediate dimension
    max_seq_len: int = 128         # Maximum sequence length
    rope_theta: float = 10000.0    # RoPE theta parameter
    norm_eps: float = 1e-6         # RMSNorm epsilon
    dropout: float = 0.0           # Dropout rate (0 for profiling)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ProfilerConfig:
    """Configuration for profiling options."""
    enable_pytorch_profiler: bool = False
    enable_deepspeed_flops: bool = False
    enable_memory_profiling: bool = False
    profile_operators: bool = False
    profile_attention_only: bool = False
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
            'gpu_utilization': [],
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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as used in LLaMA."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("rms_norm"):
            # Calculate RMS
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
            x = x / rms
            return x * self.weight


class RotaryPositionEmbedding:
    """Rotary Position Embeddings (RoPE) for better position encoding."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Cache for cos and sin
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos and sin values."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len

            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=dtype)

            # Compute frequencies
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim))
            freqs = torch.outer(t, inv_freq)

            # Cache cos and sin
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def apply_rotary_embedding(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to query and key tensors."""
        with record_function("rope_embedding"):
            batch_size, seq_len, n_heads_q, head_dim = q.shape
            _, _, n_heads_k, _ = k.shape

            self._update_cache(start_pos + seq_len, q.device, q.dtype)

            # Reshape for rotary embedding
            q = q.reshape(batch_size, seq_len, n_heads_q, head_dim // 2, 2)
            k = k.reshape(batch_size, seq_len, n_heads_k, head_dim // 2, 2)

            # Apply rotation
            cos = self._cos_cached[start_pos:start_pos + seq_len].unsqueeze(1)
            sin = self._sin_cached[start_pos:start_pos + seq_len].unsqueeze(1)

            q_rot = torch.stack([
                q[..., 0] * cos - q[..., 1] * sin,
                q[..., 0] * sin + q[..., 1] * cos
            ], dim=-1)

            k_rot = torch.stack([
                k[..., 0] * cos - k[..., 1] * sin,
                k[..., 0] * sin + k[..., 1] * cos
            ], dim=-1)

            # Reshape back
            q_rot = q_rot.reshape(batch_size, seq_len, n_heads_q, head_dim)
            k_rot = k_rot.reshape(batch_size, seq_len, n_heads_k, head_dim)

            return q_rot, k_rot


class Attention(nn.Module):
    """Multi-head attention with optional Grouped Query Attention (GQA)."""

    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections (separate for profiling analysis)
        self.q_proj = nn.Linear(config.hidden_dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.hidden_dim, bias=False)

        # Rotary embeddings
        self.rope = RotaryPositionEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, start_pos: int = 0) -> torch.Tensor:
        with record_function("attention_forward"):
            batch_size, seq_len, _ = x.shape

            # Project to Q, K, V (separate operations for profiling visibility)
            with record_function("qkv_projection"):
                q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
                k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
                v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

            # Apply rotary position embeddings
            q, k = self.rope.apply_rotary_embedding(q, k, start_pos)

            # Repeat K,V heads if using GQA
            if self.n_kv_heads < self.n_heads:
                k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
                v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

            # Transpose for attention computation
            q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Compute attention scores
            with record_function("attention_scores"):
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                # Apply causal mask
                if mask is not None:
                    scores = scores + mask

            # Compute attention weights
            with record_function("attention_softmax"):
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            with record_function("attention_output"):
                attn_output = torch.matmul(attn_weights, v)

                # Reshape and project output
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
                output = self.o_proj(attn_output)

            return output


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in LLaMA."""

    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with record_function("swiglu_forward"):
            with record_function("swiglu_gate_up"):
                gate = F.silu(self.gate_proj(x))
                up = self.up_proj(x)
                intermediate = gate * up

            with record_function("swiglu_down"):
                output = self.down_proj(intermediate)
                return self.dropout(output)


class TransformerBlock(nn.Module):
    """Single transformer block with RMSNorm and SwiGLU."""

    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = SwiGLU(config)
        self.norm1 = RMSNorm(config.hidden_dim, config.norm_eps)
        self.norm2 = RMSNorm(config.hidden_dim, config.norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with record_function("transformer_block"):
            # Pre-norm attention
            with record_function("attention_residual"):
                x = x + self.attention(self.norm1(x), mask)

            # Pre-norm feed-forward
            with record_function("ffn_residual"):
                x = x + self.feed_forward(self.norm2(x))

            return x


class TinyLlama(nn.Module):
    """Tiny LLaMA model for profiling and demonstration."""

    def __init__(self, config: TinyLlamaConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm and output projection
        self.norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights (optional but common)
        self.output_proj.weight = self.token_embedding.weight

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

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        with record_function("model_forward"):
            batch_size, seq_len = input_ids.shape

            # Create causal mask
            with record_function("causal_mask"):
                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), diagonal=1)
                mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

            # Token embeddings
            with record_function("token_embedding"):
                x = self.token_embedding(input_ids)

            # Pass through transformer blocks
            with record_function("transformer_layers"):
                for i, block in enumerate(self.blocks):
                    with record_function(f"layer_{i}"):
                        x = block(x, mask)

            # Final norm and output projection
            with record_function("final_output"):
                x = self.norm(x)
                logits = self.output_proj(x)

            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                with record_function("loss_calculation"):
                    # Shift for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Calculate cross-entropy loss
                    loss = F.cross_entropy(
                        shift_logits.view(-1, self.config.vocab_size),
                        shift_labels.view(-1)
                    )

            return {'logits': logits, 'loss': loss}


class SimpleTextDataset:
    """Simple text dataset for training demonstration."""

    def __init__(self, seq_length: int = 128, vocab_size: int = 1000, num_samples: int = 1000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_samples = num_samples

        # Generate synthetic text data (deterministic)
        np.random.seed(42)
        self.data = np.random.randint(1, vocab_size, size=(num_samples, seq_length + 1), dtype=np.int64)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""
        indices = np.random.choice(self.num_samples, batch_size, replace=False)
        batch = self.data[indices]

        # Split into input and target
        input_ids = torch.from_numpy(batch[:, :-1])
        labels = torch.from_numpy(batch[:, 1:])

        return input_ids, labels


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


def setup_deepspeed_profiler(model: nn.Module) -> Optional[FlopsProfiler]:
    """Setup DeepSpeed FLOPS profiler."""
    if not DEEPSPEED_AVAILABLE:
        return None

    return FlopsProfiler(model)


def train_tiny_llama(
    config: TinyLlamaConfig,
    profiler_config: ProfilerConfig,
    num_steps: int = 50,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    use_amp: bool = False
):
    """Train the Tiny LLaMA model with comprehensive profiling."""

    # Setup environment
    setup_deterministic_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = TinyLlama(config).to(device)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Configuration:")
    print(f"   Vocabulary size: {config.vocab_size:,}")
    print(f"   Hidden dimension: {config.hidden_dim}")
    print(f"   Number of layers: {config.n_layers}")
    print(f"   Number of heads: {config.n_heads}")
    print(f"   Sequence length: {config.max_seq_len}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1e6:.1f} MB (FP32)")

    # Create dataset
    dataset = SimpleTextDataset(
        seq_length=config.max_seq_len,
        vocab_size=config.vocab_size
    )

    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Setup mixed precision
    scaler = GradScaler() if use_amp else None

    # Setup profilers
    pytorch_profiler = setup_pytorch_profiler(profiler_config)
    deepspeed_profiler = setup_deepspeed_profiler(model) if profiler_config.enable_deepspeed_flops else None

    # Performance monitor
    monitor = PerformanceMonitor()

    print(f"\nTraining Configuration:")
    print(f"   Training steps: {num_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Mixed precision: {use_amp}")
    print(f"   Device: {device}")
    print(f"   PyTorch Profiler: {profiler_config.enable_pytorch_profiler}")
    print(f"   DeepSpeed FLOPS: {profiler_config.enable_deepspeed_flops}")
    print(f"   Memory Profiling: {profiler_config.enable_memory_profiling}")

    # Training loop
    model.train()

    # Warmup steps to eliminate compilation overhead
    warmup_steps = 5
    print(f"\nRunning {warmup_steps} warmup steps to eliminate compilation overhead...")
    model.train()

    for step in range(warmup_steps):
        input_ids, labels = dataset.get_batch(batch_size)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if use_amp:
            with autocast():
                outputs = model(input_ids, labels)
                loss = outputs['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

    print(f"Warmup complete. Starting measured training loop...")

    # Start FLOPS profiler after warmup
    if deepspeed_profiler:
        deepspeed_profiler.start_profile()

    print("=" * 70)

    for step in range(num_steps):
        # Start batch timing
        batch_timings = {}
        monitor.start_timing()

        # Get batch
        with nvtx.range("data_loading"):
            input_ids, labels = dataset.get_batch(batch_size)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

        # Forward pass timing
        monitor.start_timing()
        with nvtx.range("forward_pass"):
            if use_amp:
                with autocast():
                    outputs = model(input_ids, labels)
                    loss = outputs['loss']
            else:
                outputs = model(input_ids, labels)
                loss = outputs['loss']
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

    # Stop FLOPS profiler and get results
    if deepspeed_profiler:
        deepspeed_profiler.stop_profile()
        flops_summary = deepspeed_profiler.get_total_flops()
        params_summary = deepspeed_profiler.get_total_params()

        print(f"\nFLOPS Analysis:")
        print(f"   Total FLOPS: {flops_summary:,}")
        print(f"   Total Parameters: {params_summary:,}")
        if num_steps > 0 and batch_timings.get('total', 0) > 0:
            avg_time = np.mean(monitor.metrics['batch_times'])
            flops_per_sec = flops_summary / avg_time if avg_time > 0 else 0
            print(f"   FLOPS/sec: {flops_per_sec:.2e}")

    # Performance summary
    summary = monitor.get_summary()
    avg_speed = summary.get('avg_training_speed', 0)
    seq_len = config.max_seq_len
    tokens_per_sec = avg_speed * seq_len

    print(f"\nPerformance Summary:")
    print(f"   Total samples processed: {summary.get('total_samples', 0):,}")
    print(f"   Average training speed: {avg_speed:.1f} samples/sec")
    print(f"   Throughput: {tokens_per_sec:.0f} tokens/sec")
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
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)

        print(f"\nPerformance data saved to: {profile_path}")

    return model, monitor


def main():
    """Main entry point for Version 1 training."""
    parser = argparse.ArgumentParser(description='Tiny LLaMA V1: PyTorch Baseline with Profiling')

    # Model configuration
    parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length')

    # Training configuration
    parser.add_argument('--num-steps', type=int, default=50, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')

    # Profiling configuration
    parser.add_argument('--enable-pytorch-profiler', action='store_true', help='Enable PyTorch profiler')
    parser.add_argument('--enable-deepspeed-flops', action='store_true', help='Enable DeepSpeed FLOPS profiler')
    parser.add_argument('--enable-memory-profiling', action='store_true', help='Enable memory profiling')
    parser.add_argument('--enable-all-profiling', action='store_true', help='Enable all profiling features')
    parser.add_argument('--profile-operators', action='store_true', help='Profile individual operators')
    parser.add_argument('--profile-attention-only', action='store_true', help='Profile only attention operations')
    parser.add_argument('--profile-dir', type=str, default='./pytorch_profiles', help='Profiling output directory')
    parser.add_argument('--sort-by', type=str, default='cuda_time_total', help='Sort profiling results by metric')
    parser.add_argument('--warmup-steps', type=int, default=3, help='Profiler warmup steps')
    parser.add_argument('--profile-steps', type=int, default=5, help='Number of profiling steps')

    # Validation and debugging
    parser.add_argument('--validate-setup', action='store_true', help='Run validation checks')

    args = parser.parse_args()

    # Print banner
    print("=" * 80)
    print("CASTILLE AI WORKSHOP - VERSION 1: PYTORCH BASELINE")
    print("     Comprehensive Profiling Foundation for Transformer Optimization")
    print("=" * 80)

    # Configure model
    config = TinyLlamaConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        n_heads=args.num_heads,
        intermediate_dim=args.hidden_dim * 4,  # Standard 4x multiplier for fair comparison
        max_seq_len=args.seq_len
    )

    # Configure profiler
    profiler_config = ProfilerConfig(
        enable_pytorch_profiler=args.enable_pytorch_profiler or args.enable_all_profiling,
        enable_deepspeed_flops=args.enable_deepspeed_flops or args.enable_all_profiling,
        enable_memory_profiling=args.enable_memory_profiling or args.enable_all_profiling,
        profile_operators=args.profile_operators,
        profile_attention_only=args.profile_attention_only,
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
            model, monitor = train_tiny_llama(
                config=config,
                profiler_config=profiler_config,
                num_steps=3,
                batch_size=4
            )
            print("Validation successful! Environment ready for workshop.")
            return
        except Exception as e:
            print(f"Validation failed: {e}")
            return

    # Run training with profiling
    try:
        model, monitor = train_tiny_llama(
            config=config,
            profiler_config=profiler_config,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_amp=args.use_amp
        )

        print(f"\nTraining completed successfully!")

        if profiler_config.enable_pytorch_profiler:
            print(f"PyTorch profiling data saved to: {args.profile_dir}")
            print(f"   Launch TensorBoard: tensorboard --logdir {args.profile_dir}")

        print(f"\nNext Steps:")
        print(f"   1. Analyze profiling results to identify bottlenecks")
        print(f"   2. Review performance metrics and optimization opportunities")
        print(f"   3. Proceed to Version 2 for kernel fusion optimizations")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()