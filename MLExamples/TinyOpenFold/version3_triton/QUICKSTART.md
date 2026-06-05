# TinyOpenFold V3 Quick Start Guide

**5-Minute Setup and Run Guide**

## Prerequisites

- Python 3.8+
- PyTorch with CUDA/ROCm support
- Triton installed (`pip install triton`)
- AMD MI300X or compatible GPU

## Quick Start (3 Commands)

```bash
# 1. Navigate to version3_triton directory
cd version3_triton/

# 2. Run the model
python3 tiny_openfold_v3.py

# 3. View results
cat triton_profiles/performance_summary_v3.json
```

## Expected Output

```
========== TINY OPENFOLD - VERSION 3: TRITON CUSTOM KERNELS ==========

Model V3 Configuration:
   MSA dimension: 64
   Pair dimension: 128
   Evoformer blocks: 4
   Total parameters: 2,641,728
   Model size: 10.6 MB (FP32)

Triton Kernel Optimizations:
   layernorm: ACTIVE
   flash_attention_msa_row: ACTIVE
   flash_attention_msa_col: ACTIVE
   flash_attention_triangle: ACTIVE

Performance Summary V3:
   Average training speed: 150-200 samples/sec
   Peak memory usage: 80-100 MB
```

## Common Commands

### Run with Custom Parameters

```bash
# Larger batch size
python3 tiny_openfold_v3.py --batch-size 8 --num-steps 100

# Different model size
python3 tiny_openfold_v3.py --msa-dim 128 --pair-dim 256

# Longer sequence
python3 tiny_openfold_v3.py --seq-len 128
```

### Test Correctness

```bash
python3 test_correctness.py
```

### Profile Performance

```bash
# Detailed profiling
python3 run_triton_profiling.py

# Results in: profiling_results/
```

### Compare All Versions

```bash
# Run comprehensive comparison
./launch_performance_study.sh

# Results in: performance_study_TIMESTAMP/
```

### Hardware Profiling (ROCm)

```bash
./run_rocprof_triton.sh

# Results in: rocprof_results_v3/
```

## Configuration Options

```bash
python3 tiny_openfold_v3.py --help
```

**Key Parameters**:
- `--batch-size`: Batch size (default: 4)
- `--num-steps`: Training steps (default: 50)
- `--seq-len`: Sequence length (default: 64)
- `--num-blocks`: Evoformer blocks (default: 4)
- `--msa-dim`: MSA dimension (default: 64)
- `--pair-dim`: Pair dimension (default: 128)

## Troubleshooting

### "Triton not found"

```bash
pip install triton
```

### "CUDA out of memory"

```bash
# Reduce batch size or sequence length
python3 tiny_openfold_v3.py --batch-size 2 --seq-len 32
```

### "Import Error"

```bash
# Make sure you're in the correct directory
cd /path/to/TinyOpenFold/version3_triton/
```

## Learning Path

1. **Quick Test** (5 min): Run default training
2. **Understand Code** (30 min): Read through tiny_openfold_v3.py
3. **Exercise 1** (45 min): Learn Triton basics
4. **Exercise 2** (60 min): Triangle optimization
5. **Exercise 3** (75 min): Flash Attention

## File Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `tiny_openfold_v3.py` | Main model | Training and inference |
| `test_correctness.py` | Verify implementation | After changes |
| `run_triton_profiling.py` | Benchmark kernels | Performance analysis |
| `launch_performance_study.sh` | Compare versions | V1 vs V2 vs V3 |
| `README.md` | Full documentation | Deep dive |

## Performance Expectations

For default configuration (batch=4, seq_len=64):

| Version | Speed (samples/s) | Memory (MB) |
|---------|-------------------|-------------|
| V1 (Baseline) | ~75 | ~196 |
| V2 (Fused) | ~110-120 | ~120-140 |
| V3 (Triton) | **~150-200** | **~80-100** |

**V3 Speedup**: 2.0-2.7x faster than V1  
**V3 Memory**: 50-60% reduction vs V1

## Next Steps

After successful run:

1. ✅ Check `triton_profiles/performance_summary_v3.json`
2. 📊 Compare with V1/V2 using `launch_performance_study.sh`
3. 🔬 Profile with `run_triton_profiling.py`
4. 🚀 Experiment with different configurations

## Support

- **Full Documentation**: `README.md`
- **Architecture**: `../ARCHITECTURE.md`
- **Main Tutorial**: `../PERFORMANCE_OPTIMIZATION_TUTORIAL.md`

## Quick Links

- [Full README](README.md)
- [Architecture Details](../ARCHITECTURE.md)
- [Optimization Tutorial](../PERFORMANCE_OPTIMIZATION_TUTORIAL.md)

---

**Ready to start?** Run: `python3 tiny_openfold_v3.py`

