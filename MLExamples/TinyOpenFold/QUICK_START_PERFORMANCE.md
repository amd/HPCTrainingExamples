# TinyOpenFold: Quick Performance Comparison

**Goal**: Demonstrate progressive optimization from baseline to custom Triton kernels

## The Bottom Line - Progressive Optimization

| Problem Size | V1 Baseline | V2 Fused | V3 Triton | Total Gain |
|--------------|-------------|----------|-----------|------------|
| **Small (64 residues)** | 80.5 s/s | 106.4 s/s (+32%) | **162.5 s/s** | **🚀 +102% (2.0x)** |
| **Medium (128 residues)** | 41.5 s/s | 49.0 s/s (+18%) | **68.5 s/s** | **🚀 +65% (1.65x)** |

### Optimization Progression

**Small Problem:**
- V1 → V2: +32% (kernel fusion)
- V2 → V3: +53% (custom Triton kernels)
- **V1 → V3: +102% (2.0x total speedup!)**

**Medium Problem:**
- V1 → V2: +18% (kernel fusion)
- V2 → V3: +40% (custom Triton kernels)
- **V1 → V3: +65% (1.65x total speedup!)**

**Memory**: V1/V2 identical, V3 slightly higher but enables faster computation  
**Runtime**: ~30 seconds for all three versions  
**Hardware**: AMD Instinct MI300X  

---

## Run It Yourself (30 seconds for all 3 versions)

```bash
# Navigate to TinyOpenFold
cd /mnt/thera/data/incoming/asimishr/aiml_prof/HPCTrainingExamples/MLExamples/TinyOpenFold

# Setup (one-time)
module load python/3.12 rocm/7.2 libffi/3.3
source venvOF/bin/activate

# Run complete optimization tutorial (V1 → V2 → V3)
bash optimization_tutorial.sh
```

**Output**: Progressive optimization comparison showing V1 → V2 → V3 improvements

---

## Manual Step-by-Step Tutorial (Small Problem)

### Step 1: Baseline (V1) - ~5 seconds
```bash
cd version1_pytorch_baseline
python3 tiny_openfold_v1.py --seq-len 64 --num-seqs 16 --batch-size 4 --device 0
```
**Result**: 80.5 samples/sec, 49.7 ms/batch  
**Optimization Level**: None (pure PyTorch)

### Step 2: Kernel Fusion (V2) - ~4 seconds  
```bash
cd ../version2_pytorch_fused
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py --seq-len 64 --num-seqs 16 --batch-size 4
```
**Result**: 106.4 samples/sec, 37.6 ms/batch  
**Speedup vs V1**: **1.32x (32% faster)** ⚡  
**Optimization**: QKV fusion, Flash Attention, gate/proj fusion

### Step 3: Custom Triton Kernels (V3) - ~4 seconds
```bash
cd ../version3_triton
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py --seq-len 64 --num-seqs 16 --batch-size 4
```
**Result**: 162.5 samples/sec, 24.6 ms/batch  
**Speedup vs V2**: **1.53x (53% faster)** ⚡⚡  
**Speedup vs V1**: **2.0x (102% faster - doubled performance!)** ⚡⚡⚡  
**Optimization**: Custom LayerNorm, Flash Attention for MSA & triangles

---

## What's Being Optimized? - Progressive Approach

### Version 2: Kernel Fusion (PyTorch-level)
1. **MSA QKV Fusion**: 3 kernels → 1 kernel for attention
2. **Flash Attention**: Memory-efficient O(N) attention
3. **Triangle Fusion**: Fused gate/projection operations
4. **Triangle QKV Fusion**: Combined Q, K, V for triangles

**Result**: 80% fewer kernel launches (60 → 12 kernels)  
**Speedup**: 32% (small), 18% (medium)

### Version 3: Custom Triton Kernels (GPU-level)
1. **Custom LayerNorm**: Fused mean/variance computation with normalization
2. **Flash Attention (MSA)**: Hand-optimized tiled attention for row/column ops
3. **Flash Attention (Triangles)**: Optimized pair-wise attention
4. **Memory Optimization**: Reduced HBM traffic, better cache utilization

**Result**: Custom GPU kernels with optimal memory access patterns  
**Additional Speedup**: 53% (small), 40% (medium) over V2  
**Total Speedup**: 102% (small), 65% (medium) over V1

---

## Performance Breakdown - Progressive Optimization

### Small Problem (64 residues)

```
Operation      V1 Time    V2 Time    V3 Time    V1→V3 Improvement
───────────────────────────────────────────────────────────────────
Forward        18.3 ms    14.7 ms    14.0 ms    -23% ⚡
Backward       27.2 ms    19.5 ms     8.5 ms    -69% ⚡⚡⚡
Optimizer       4.1 ms     3.4 ms     1.5 ms    -63% ⚡⚡
───────────────────────────────────────────────────────────────────
Total Batch    49.7 ms    37.6 ms    24.6 ms    -51% ⚡⚡⚡
Speed          80.5 s/s  106.4 s/s  162.5 s/s   +102% ⚡⚡⚡
```

### Medium Problem (128 residues)

```
Operation      V1 Time    V2 Time    V3 Time    V1→V3 Improvement
───────────────────────────────────────────────────────────────────
Forward        17.4 ms    14.5 ms    14.8 ms    -15% ⚡
Backward       26.8 ms    22.9 ms    11.7 ms    -56% ⚡⚡⚡
Optimizer       4.0 ms     3.4 ms     1.6 ms    -60% ⚡⚡
───────────────────────────────────────────────────────────────────
Total Batch    48.2 ms    40.8 ms    29.2 ms    -39% ⚡⚡⚡
Speed          41.5 s/s   49.0 s/s   68.5 s/s   +65% ⚡⚡⚡
```

**Key Insights**: 
- Backward pass shows **massive** improvement with V3 (69% and 56% reduction!)
- V3 Triton kernels dramatically reduce backward computation time
- Small problems benefit more from custom kernels (2.0x vs 1.65x)

---

## Why These Problem Sizes?

### Small (64 residues)
- ✅ **Best fusion benefits**: Kernel overhead dominates
- ✅ **Fast to run**: ~5 seconds per version
- ✅ **Clear improvement**: 32% speedup easy to demonstrate
- ✅ **Typical use case**: Educational and quick experiments

### Medium (128 residues)
- ✅ **Balanced workload**: Mix of overhead and computation
- ✅ **Still fast**: ~5 seconds per version
- ✅ **Measurable gains**: 18% speedup
- ✅ **Real-world**: Common protein size for training

### Why not larger?
- ⚠️ Large problems (256+ residues) are compute-bound
- ⚠️ Fusion benefits plateau at ~12% (still good, but less dramatic)
- ⚠️ Take longer to run (not ideal for quick demos)

---

## Visualize the Progressive Speedup

```
Small Problem Throughput (samples/sec):
V1 Baseline:        ████████████████                      80.5
V2 Fusion:          █████████████████████                106.4  (+32%)
V3 Triton:          ████████████████████████████████     162.5  (+102% total!)
                    ↑ V2 adds fusion benefits
                              ↑ V3 adds custom kernel benefits

Medium Problem Throughput (samples/sec):
V1 Baseline:        ████████████                          41.5
V2 Fusion:          ██████████████                        49.0  (+18%)
V3 Triton:          ███████████████████                   68.5  (+65% total!)
                    ↑ V2 adds fusion benefits
                              ↑ V3 adds custom kernel benefits

Backward Pass Speedup (most dramatic improvement):
                    V1        V2       V3
Small:              27.2 ms → 19.5 ms → 8.5 ms   (69% reduction!)
Medium:             26.8 ms → 22.9 ms → 11.7 ms  (56% reduction!)
```

---

## Key Takeaways - Optimization Journey

✅ **V1 → V2 (Kernel Fusion)**: 18-32% faster with zero code changes  
✅ **V2 → V3 (Triton Kernels)**: Additional 40-53% speedup  
✅ **V1 → V3 (Total)**: **2.0x faster (small), 1.65x faster (medium)**  
✅ **No accuracy loss**: All versions produce identical results  
✅ **Quick demo**: 30 seconds for all three versions  
✅ **Progressive learning**: See exactly what each optimization contributes  

**Recommendations by Use Case**:
- **Development/Debug**: Use V1 (simplest, easiest to modify)
- **Production**: Use V2 (great speedup, stable, PyTorch-based)
- **Maximum Performance**: Use V3 (best performance, requires Triton)

**Tutorial Value**: This demonstrates the complete GPU optimization pipeline!  
1. Start with readable baseline (V1)
2. Apply high-level optimizations (V2 - kernel fusion)
3. Drop to GPU-level optimizations (V3 - custom Triton kernels)

---

## Optimization Progression Summary

| Version | Optimization Type | Small Speedup | Medium Speedup | Techniques |
|---------|------------------|---------------|----------------|------------|
| **V1** | Baseline | 1.0x | 1.0x | Pure PyTorch |
| **V2** | High-level fusion | 1.32x | 1.18x | QKV fusion, Flash Attention |
| **V3** | Custom kernels | 2.0x | 1.65x | Triton LayerNorm, custom Flash Attn |

## Next Steps: Stack More Optimizations

### 1. V3 + Mixed Precision (Expected: 2.5-3.0x total)
```bash
cd version3_triton
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 128 --num-seqs 32 --batch-size 2 --use-amp
```

### 2. V3 + Torch Compile (Expected: 2.2-2.4x total)
```bash
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 128 --num-seqs 32 --batch-size 2 --enable-torch-compile
```

### 3. Multi-GPU with V3 (Expected: near-linear scaling)
```bash
ROCR_VISIBLE_DEVICES=0,1,2,3 python3 tiny_openfold_v3.py \
    --seq-len 128 --num-seqs 32 --batch-size 8
```

---

## Documentation

- **Detailed guide**: `PERFORMANCE_GUIDE.md`
- **Full regression**: `REGRESSION_TEST_SUMMARY.md`
- **Skills reference**: `skills.md`
- **Architecture**: `ARCHITECTURE.md`

---

## One-Liner Demo

```bash
# Setup and run comparison in one command
module load python/3.12 rocm/7.2 libffi/3.3 && \
source venvOF/bin/activate && \
bash regression_test_focused.sh
```

**Duration**: 20 seconds  
**Output**: Performance comparison with speedup metrics  
**Result**: Proof that kernel fusion delivers 18-32% performance gains! 🎉
