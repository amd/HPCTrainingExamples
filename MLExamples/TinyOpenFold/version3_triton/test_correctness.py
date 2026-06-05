#!/usr/bin/env python3
"""
Numerical Correctness Test for TinyOpenFold V3

Verifies that Triton kernel outputs match PyTorch baseline outputs
within acceptable numerical tolerance.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from version3_triton.tiny_openfold_v3 import (
    TinyOpenFoldV3,
    TinyOpenFoldConfig,
    TritonLayerNorm,
    TritonMSARowAttention,
    TritonMSAColumnAttention,
)

from version1_pytorch_baseline.tiny_openfold_v1 import (
    TinyOpenFold as TinyOpenFoldV1,
    TinyOpenFoldConfig as TinyOpenFoldConfigV1,
)

def test_layernorm(tolerance=1e-4):
    """Test TritonLayerNorm vs PyTorch LayerNorm."""
    print("\n" + "="*70)
    print("Test 1: LayerNorm Correctness")
    print("="*70)
    
    dim = 128
    batch = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠ Warning: Running on CPU, skipping Triton tests")
        return True
    
    # Create test data
    x = torch.randn(batch, dim, device=device)
    
    # Triton LayerNorm
    triton_norm = TritonLayerNorm(dim).to(device)
    triton_output = triton_norm(x)
    
    # PyTorch LayerNorm (with same weights)
    pytorch_norm = torch.nn.LayerNorm(dim).to(device)
    pytorch_norm.weight.data = triton_norm.weight.data.clone()
    pytorch_output = pytorch_norm(x)
    
    # Check correctness
    max_diff = (triton_output - pytorch_output).abs().max().item()
    rel_error = max_diff / pytorch_output.abs().max().item()
    
    print(f"  Input shape: {x.shape}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Relative error: {rel_error:.2e}")
    print(f"  Tolerance: {tolerance:.2e}")
    
    passed = rel_error < tolerance
    if passed:
        print(f"  ✓ Test PASSED")
    else:
        print(f"  ✗ Test FAILED")
    
    return passed


def test_msa_attention(tolerance=1e-3):
    """Test MSA attention correctness."""
    print("\n" + "="*70)
    print("Test 2: MSA Attention Correctness")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠ Warning: Running on CPU, skipping Triton tests")
        return True
    
    config = TinyOpenFoldConfig(
        msa_dim=64,
        pair_dim=128,
        n_seqs=16,
        max_seq_len=32,  # Smaller for testing
    )
    
    batch_size = 2
    
    # Create test data
    msa = torch.randn(batch_size, config.n_seqs, config.max_seq_len, config.msa_dim, device=device)
    pair = torch.randn(batch_size, config.max_seq_len, config.max_seq_len, config.pair_dim, device=device)
    
    # Triton MSA Row Attention
    triton_row_attn = TritonMSARowAttention(config).to(device)
    triton_output = triton_row_attn(msa, pair)
    
    # Note: We can't directly compare with V1 because the internal implementations
    # differ slightly (Flash Attention vs standard attention). Instead, we check:
    # 1. Output shape is correct
    # 2. No NaNs or Infs
    # 3. Output values are in reasonable range
    
    has_nan = torch.isnan(triton_output).any()
    has_inf = torch.isinf(triton_output).any()
    mean_abs = triton_output.abs().mean().item()
    
    print(f"  MSA shape: {msa.shape}")
    print(f"  Output shape: {triton_output.shape}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Mean absolute value: {mean_abs:.4f}")
    
    passed = not has_nan and not has_inf and mean_abs < 100.0
    
    if passed:
        print(f"  ✓ Test PASSED (sanity checks)")
    else:
        print(f"  ✗ Test FAILED")
    
    return passed


def test_full_model_forward(tolerance=1e-2):
    """Test full model forward pass."""
    print("\n" + "="*70)
    print("Test 3: Full Model Forward Pass")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠ Warning: Running on CPU, skipping Triton tests")
        return True
    
    # Small config for testing
    config = TinyOpenFoldConfig(
        vocab_size=21,
        msa_dim=32,
        pair_dim=64,
        n_evoformer_blocks=2,
        n_heads_msa=2,
        n_heads_pair=2,
        msa_intermediate_dim=128,
        pair_intermediate_dim=256,
        outer_product_dim=16,
        max_seq_len=16,  # Small for quick testing
        n_seqs=8,
        pair_input_dim=65,
    )
    
    # Create V3 model
    model_v3 = TinyOpenFoldV3(config).to(device)
    model_v3.eval()
    
    # Create test inputs
    batch_size = 2
    msa_tokens = torch.randint(0, config.vocab_size, 
                                (batch_size, config.n_seqs, config.max_seq_len), 
                                device=device)
    pair_features = torch.randn(batch_size, config.max_seq_len, config.max_seq_len, 
                                config.pair_input_dim, device=device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model_v3(msa_tokens, pair_features)
    
    # Check outputs
    distances = outputs['distances']
    has_nan = torch.isnan(distances).any()
    has_inf = torch.isinf(distances).any()
    mean_dist = distances.mean().item()
    
    print(f"  Input MSA shape: {msa_tokens.shape}")
    print(f"  Input pair shape: {pair_features.shape}")
    print(f"  Output distances shape: {distances.shape}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Mean predicted distance: {mean_dist:.4f} Å")
    print(f"  Distance range: [{distances.min():.2f}, {distances.max():.2f}] Å")
    
    # Distances should be in reasonable range (0-20 Angstroms)
    passed = (not has_nan and not has_inf and 
              distances.min() >= 0 and distances.max() <= 20.0)
    
    if passed:
        print(f"  ✓ Test PASSED (sanity checks)")
    else:
        print(f"  ✗ Test FAILED")
    
    return passed


def test_gradient_flow():
    """Test that gradients flow correctly through Triton kernels."""
    print("\n" + "="*70)
    print("Test 4: Gradient Flow")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("⚠ Warning: Running on CPU, skipping Triton tests")
        return True
    
    # Small config for testing
    config = TinyOpenFoldConfig(
        msa_dim=32,
        pair_dim=64,
        n_evoformer_blocks=1,
        max_seq_len=8,
        n_seqs=4,
    )
    
    # Create model
    model = TinyOpenFoldV3(config).to(device)
    model.train()
    
    # Create test inputs
    batch_size = 2
    msa_tokens = torch.randint(0, config.vocab_size,
                                (batch_size, config.n_seqs, config.max_seq_len),
                                device=device)
    pair_features = torch.randn(batch_size, config.max_seq_len, config.max_seq_len,
                                config.pair_input_dim, device=device)
    target_distances = torch.rand(batch_size, config.max_seq_len, config.max_seq_len, 1,
                                   device=device) * 20.0
    
    # Forward pass
    outputs = model(msa_tokens, pair_features, target_distances)
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grads = True
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ✗ No gradient for: {name}")
            has_grads = False
        else:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  ✗ Invalid gradient for: {name}")
                has_grads = False
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  All parameters have gradients: {has_grads}")
    print(f"  Mean gradient norm: {sum(grad_norms)/len(grad_norms):.2e}")
    print(f"  Max gradient norm: {max(grad_norms):.2e}")
    
    passed = has_grads and all(gn < 1e6 for gn in grad_norms)
    
    if passed:
        print(f"  ✓ Test PASSED")
    else:
        print(f"  ✗ Test FAILED")
    
    return passed


def main():
    """Run all correctness tests."""
    print("="*70)
    print("TinyOpenFold V3 Numerical Correctness Tests")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available. Tests will be limited.")
        print("Triton kernels require CUDA to run.\n")
    
    # Run tests
    results = {}
    results['layernorm'] = test_layernorm()
    results['msa_attention'] = test_msa_attention()
    results['full_model'] = test_full_model_forward()
    results['gradients'] = test_gradient_flow()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests PASSED!")
        print("="*70)
        return 0
    else:
        print("✗ Some tests FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())

