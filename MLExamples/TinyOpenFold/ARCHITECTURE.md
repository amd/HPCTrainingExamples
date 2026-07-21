# TinyOpenFold Architecture Documentation

**Source File**: `HPCTrainingExamples/MLExamples/TinyOpenFold/version1_pytorch_baseline/tiny_openfold_v1.py`

## Overview

TinyOpenFold is a simplified, educational implementation of the AlphaFold 2 architecture, focusing on the core innovation: the **Evoformer**. This implementation demonstrates how Multiple Sequence Alignments (MSA) and pairwise residue representations interact to predict protein structures.

## Core Architecture: The Evoformer

The Evoformer is the main building block of AlphaFold 2, processing two coupled representations:
1. **MSA Representation** (N_seq × N_res × msa_dim): Features for each residue in each sequence
2. **Pair Representation** (N_res × N_res × pair_dim): Pairwise features between residues

These representations are updated through a series of attention and communication operations.

## Architecture Components

### 1. Input Embeddings

#### MSA Embedding
**Shape**: `(batch, n_seqs, seq_len, msa_dim)`

Maps discrete amino acid tokens in the MSA to continuous vectors.

**Parameters**: `vocab_size × msa_dim`
- Example (TinyOpenFoldConfig): 21 amino acids × 64 dim = **1,344 parameters**

#### Pair Embedding
**Shape**: `(batch, seq_len, seq_len, pair_dim)`

Encodes pairwise information between residues (e.g., distance bins, relative positions).

**Parameters**: `pair_input_dim × pair_dim`
- Example (TinyOpenFoldConfig): 65 features × 128 dim = **8,320 parameters**

### 2. Evoformer Block (Repeated n_evoformer_blocks times)

Each Evoformer block contains multiple sub-modules that update both MSA and pair representations.

#### A. MSA Row-wise Attention with Pair Bias

Attention across residues within each MSA sequence, biased by pair representation.

**MSA Attention Components**:
- Query projection: `(msa_dim, n_heads_msa × head_dim_msa)`
- Key projection: `(msa_dim, n_heads_msa × head_dim_msa)`
- Value projection: `(msa_dim, n_heads_msa × head_dim_msa)`
- Output projection: `(n_heads_msa × head_dim_msa, msa_dim)`
- Pair bias projection: `(pair_dim, n_heads_msa)`

**Total MSA Row Attention Parameters**:
```
3 × msa_dim × (n_heads_msa × head_dim_msa) + (n_heads_msa × head_dim_msa) × msa_dim + pair_dim × n_heads_msa
= 4 × msa_dim² + pair_dim × n_heads_msa
```

Example (msa_dim=64, n_heads_msa=4, pair_dim=128):
- Q, K, V, O: 4 × 64² = 16,384
- Pair bias: 128 × 4 = 512
- **Total: 16,896 parameters**

#### B. MSA Column-wise Attention

Attention across sequences for each residue position (communication between different sequences).

**Parameters**: Same structure as row attention but without pair bias
```
4 × msa_dim²
```

Example (msa_dim=64):
- **Total: 16,384 parameters**

#### C. MSA Transition (Feed-Forward)

Per-position feed-forward network for MSA representation.

**Layers**:
- Linear 1: `(msa_dim, msa_intermediate_dim)`
- Linear 2: `(msa_intermediate_dim, msa_dim)`

**Total MSA Transition Parameters**:
```
2 × msa_dim × msa_intermediate_dim
```

Example (msa_dim=64, msa_intermediate_dim=256):
- **Total: 32,768 parameters**

#### D. Outer Product Mean

Projects MSA representation to update pair representation using outer product.

**Layers**:
- MSA to outer: `(msa_dim, outer_product_dim)`
- Outer to pair: `(outer_product_dim², pair_dim)`

**Total Outer Product Parameters**:
```
msa_dim × outer_product_dim + outer_product_dim² × pair_dim
```

Example (msa_dim=64, outer_product_dim=32, pair_dim=128):
- MSA projection: 64 × 32 = 2,048
- Outer to pair: 32² × 128 = 131,072
- **Total: 133,120 parameters**

#### E. Triangle Multiplicative Update (Outgoing)

Updates pair representation using geometric reasoning: if residues i-j and j-k are close, then i-k should also be considered.

**Layers**:
- Left projection: `(pair_dim, pair_dim)`
- Right projection: `(pair_dim, pair_dim)`
- Left gate: `(pair_dim, pair_dim)`
- Right gate: `(pair_dim, pair_dim)`
- Output projection: `(pair_dim, pair_dim)`
- Output gate: `(pair_dim, pair_dim)`

**Total Triangle Mult Parameters**:
```
6 × pair_dim²
```

Example (pair_dim=128):
- **Total: 98,304 parameters**

#### F. Triangle Multiplicative Update (Incoming)

Similar to outgoing but with different edge orientation.

Example (pair_dim=128):
- **Total: 98,304 parameters**

#### G. Triangle Self-Attention (Starting)

Self-attention around edges starting from a node.

**Components**:
- Q, K, V projections: `3 × pair_dim × (n_heads_pair × head_dim_pair)`
- Output projection: `(n_heads_pair × head_dim_pair, pair_dim)`

**Total Parameters**:
```
4 × pair_dim²
```

Example (pair_dim=128):
- **Total: 65,536 parameters**

#### H. Triangle Self-Attention (Ending)

Self-attention around edges ending at a node.

Example (pair_dim=128):
- **Total: 65,536 parameters**

#### I. Pair Transition (Feed-Forward)

Per-position feed-forward for pair representation.

**Total Parameters**:
```
2 × pair_dim × pair_intermediate_dim
```

Example (pair_dim=128, pair_intermediate_dim=512):
- **Total: 131,072 parameters**

#### Per Evoformer Block Total

Sum of all components:
- MSA Row Attention: 16,896
- MSA Column Attention: 16,384
- MSA Transition: 32,768
- Outer Product Mean: 133,120
- Triangle Mult (Out): 98,304
- Triangle Mult (In): 98,304
- Triangle Attn (Start): 65,536
- Triangle Attn (End): 65,536
- Pair Transition: 131,072
- **Per Block: ~658,000 parameters**

### 3. Structure Module (Simplified)

Converts pair representation to 3D coordinates.

**Simplified Version** (no IPA, direct prediction):
- Pair to distance: `(pair_dim, 1)`
- Angle predictions: `(pair_dim, 2)` (phi, psi angles)

**Parameters**: `pair_dim × 3`

Example (pair_dim=128):
- **Total: 384 parameters**

## Complete Parameter Formula

**Total Parameters** = 
```
MSA_Embedding + Pair_Embedding 
+ (n_evoformer_blocks × Per_Block_Parameters)
+ Structure_Module

= vocab_size × msa_dim
  + pair_input_dim × pair_dim
  + n_evoformer_blocks × [
      (4 × msa_dim² + pair_dim × n_heads_msa)      # MSA Row Attn
      + 4 × msa_dim²                                 # MSA Col Attn
      + 2 × msa_dim × msa_intermediate_dim          # MSA Transition
      + (msa_dim × outer_dim + outer_dim² × pair_dim) # Outer Product
      + 6 × pair_dim²                                # Triangle Mult Out
      + 6 × pair_dim²                                # Triangle Mult In
      + 4 × pair_dim²                                # Triangle Attn Start
      + 4 × pair_dim²                                # Triangle Attn End
      + 2 × pair_dim × pair_intermediate_dim         # Pair Transition
    ]
  + pair_dim × 3                                     # Structure Module
```

## Example Calculation (TinyOpenFoldConfig Default)

**Configuration**:
- `vocab_size` = 21 (20 amino acids + unknown)
- `msa_dim` = 64
- `pair_dim` = 128
- `n_evoformer_blocks` = 4
- `n_heads_msa` = 4
- `n_heads_pair` = 4
- `head_dim_msa` = 16 (msa_dim / n_heads_msa)
- `head_dim_pair` = 32 (pair_dim / n_heads_pair)
- `msa_intermediate_dim` = 256
- `pair_intermediate_dim` = 512
- `outer_product_dim` = 32
- `pair_input_dim` = 65
- `max_seq_len` = 64
- `n_seqs` = 16

**Component Breakdown**:

1. **MSA Embedding**: 21 × 64 = **1,344**

2. **Pair Embedding**: 65 × 128 = **8,320**

3. **Per Evoformer Block**:
   - MSA Row Attention: 4 × 64² + 128 × 4 = 16,896
   - MSA Column Attention: 4 × 64² = 16,384
   - MSA Transition: 2 × 64 × 256 = 32,768
   - Outer Product Mean: 64 × 32 + 32² × 128 = 133,120
   - Triangle Mult (Out): 6 × 128² = 98,304
   - Triangle Mult (In): 6 × 128² = 98,304
   - Triangle Attn (Start): 4 × 128² = 65,536
   - Triangle Attn (End): 4 × 128² = 65,536
   - Pair Transition: 2 × 128 × 512 = 131,072
   - **Subtotal per block**: 657,920

4. **All 4 Blocks**: 4 × 657,920 = **2,631,680**

5. **Structure Module**: 128 × 3 = **384**

**Total**: 1,344 + 8,320 + 2,631,680 + 384 = **2,641,728 parameters**

**Model Size**:
- FP32: 2,641,728 × 4 / 1e6 = **10.6 MB**
- FP16/BF16: 2,641,728 × 2 / 1e6 = **5.3 MB**

## Data Structure and Batching

### Batch Size
**Batch size** refers to the number of protein samples processed simultaneously in one forward/backward pass. For example, `batch_size=4` means 4 complete protein structures are processed together.

### Sample Structure
Each **sample** represents one complete protein structure with three components:

1. **MSA Tokens**: Shape `(n_seqs, seq_len)` = `(16, 64)`
   - Integer tokens (0-20) representing amino acids
   - 16 MSA sequences × 64 amino acids per sequence

2. **Pair Features**: Shape `(seq_len, seq_len, pair_input_dim)` = `(64, 64, 65)`
   - Pairwise feature matrix: 64×64 residues with 65 features per pair

3. **Target Distances**: Shape `(seq_len, seq_len, 1)` = `(64, 64, 1)`
   - Ground truth distance matrix for structure prediction

**Total per sample**: ~271K elements (mostly from pair features: 266K floats)

**Batch processing**: With `batch_size=4`, tensors have shape `(4, ...)` for all three components, enabling parallel processing of multiple proteins.

### Sample Speed Evaluation
**Training speed** (samples/sec) measures throughput and is calculated as:

```
speed = batch_size / batch_time
```

Where `batch_time` includes:
- Forward pass (model inference)
- Backward pass (gradient computation)
- Optimizer step (parameter update)

**Example**: With `batch_size=4` and `batch_time=25ms`:
- Speed = 4 / 0.025 = **160 samples/sec**

**Average training speed** is computed across all training steps, providing a stable metric for performance comparison. Higher values indicate better GPU utilization and faster training.

## Training Memory Requirements

Similar to transformers, training requires:

### Optimizer States (Adam/AdamW)
- **First Moment (m)**: Same size as parameters
- **Second Moment (v)**: Same size as parameters
- **Total**: 2× parameter memory

### Gradients
- **One gradient per parameter**: Same size as parameters

### Activations
- MSA activations: `batch × n_seqs × seq_len × msa_dim`
- Pair activations: `batch × seq_len × seq_len × pair_dim`
- Attention matrices: `batch × n_heads × seq_len × seq_len` (or `n_seqs × seq_len`)
- Typically **dominant memory consumer** for long sequences

### Total Training Memory (Approximate)
```
Total ≈ Model + Gradients + Optimizer States + Activations
     ≈ Params + Params + 2×Params + Activations
     ≈ 4×Params + Activations
```

For FP32 training with TinyOpenFoldConfig:
- Model: 10.6 MB
- Gradients: 10.6 MB
- Optimizer: 21.2 MB
- **Base**: 42.4 MB (before activations)

For batch=4, n_seqs=16, seq_len=64:
- MSA activations: 4 × 16 × 64 × 64 × 4 bytes ≈ 1 MB
- Pair activations: 4 × 64 × 64 × 128 × 4 bytes ≈ 8 MB
- Total with activations: ~50-60 MB

## Key Differences from Standard AlphaFold 2

1. **Reduced Dimensions**: 64/128 vs 256/128 in production
2. **Fewer Blocks**: 4 vs 48 Evoformer blocks
3. **No Templates**: Skips template featurization and template embedder
4. **Simplified Structure Module**: Direct distance/angle prediction instead of full IPA with frames
5. **No Recycling**: Single forward pass instead of multiple recycling iterations
6. **Synthetic Data**: Uses random MSA/pair features instead of real protein data
7. **Educational Focus**: Emphasis on clarity and understanding over production performance

## Key Innovations of Evoformer

1. **Dual Representation Updates**: MSA and pair representations evolve together, sharing information
2. **Triangle Multiplicative Updates**: Geometric inductive bias for spatial reasoning
3. **Outer Product Mean**: Projects MSA patterns onto pairwise space
4. **Pair Bias in MSA Attention**: Pairwise information guides sequence-level attention
5. **Multi-Scale Attention**: Row-wise (within sequence) and column-wise (across sequences)

## Computational Complexity

### MSA Operations
- **Row Attention**: O(n_seqs × seq_len² × msa_dim)
- **Column Attention**: O(seq_len × n_seqs² × msa_dim)
- For small MSAs, row attention dominates

### Pair Operations
- **Triangle Updates**: O(seq_len³ × pair_dim) - most expensive!
- **Triangle Attention**: O(seq_len³ × pair_dim)
- **Pair Transition**: O(seq_len² × pair_dim × pair_intermediate_dim)

### Bottlenecks
For typical configs (seq_len=64-256):
1. **Triangle operations** are O(N³) and dominate for longer sequences
2. **Pair transition** is memory-bound for large pair_dim
3. **MSA column attention** can be expensive for large MSAs

## Code Reference

```python
# From tiny_openfold_v1.py
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Model size: {total_params * 4 / 1e6:.1f} MB (FP32)")
```

## References

1. **AlphaFold 2 Paper**: Jumper et al., "Highly accurate protein structure prediction with AlphaFold", Nature 2021
2. **OpenFold**: https://github.com/aqlaboratory/openfold - Open source reproduction
3. **Evoformer Details**: AlphaFold 2 Supplement, Section 1.6
4. **Triangle Updates**: Supplement Section 1.6.7-1.6.8
5. **Structure Module**: Supplement Section 1.8

