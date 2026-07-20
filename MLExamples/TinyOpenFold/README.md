# TinyOpenFold: Educational AlphaFold 2 Implementation

A simplified, educational implementation of the AlphaFold 2 / Evoformer architecture for protein structure prediction, designed for learning and profiling.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
</p>

## Overview

TinyOpenFold is an educational implementation of the core AlphaFold 2 architecture, focusing on the **Evoformer** - the main innovation that revolutionized protein structure prediction. This implementation is designed to:

- **Teach** the fundamental concepts of AlphaFold 2's architecture
- **Profile** performance characteristics of protein structure prediction models
- **Demonstrate** how MSA (Multiple Sequence Alignment) and pair representations interact
- **Provide** a foundation for experimenting with optimization techniques

## Features

✅ **Complete Evoformer Implementation**
- MSA row-wise attention with pair bias
- MSA column-wise attention
- Triangle multiplicative updates (outgoing/incoming)
- Triangle self-attention (starting/ending)
- Outer product mean

✅ **Comprehensive Profiling Integration**
- PyTorch Profiler with GPU/CPU timeline analysis
- Memory profiling and tracking
- Operator-level performance characterization
- TensorBoard visualization support

✅ **Educational Focus**
- Clear, readable code with extensive documentation
- Parameter counting and memory analysis
- Synthetic data generation for demonstration
- Deterministic execution for reproducibility

## Quick Start

### Environment Setup and Installation

Set up your Python environment and install dependencies:

```bash
# Load modules (choose one option)
module load python/3.12 rocm/7.2        # Standard Python (recommended)
# OR
module load cray-python rocm/7.2        # Cray environment

# Navigate to TinyOpenFold directory
cd HPCTrainingExamples/MLExamples/TinyOpenFold

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Verify Python version
python3 --version

# Upgrade pip and install build tools
pip3 install --upgrade pip setuptools wheel

# Install PyTorch with ROCm support (using ROCm 7.1 nightly build)
# For ROCm 6.4:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# For ROCm 7.1 nightly (recommended):
pip3 uninstall -y torch torchvision triton torchaudio 2>/dev/null || true
pip3 install torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/nightly/rocm7.1

# Fix libcaffe2_nvrtc.so library loading issue
# Ensure ROCm and libffi modules are loaded (sets up library paths)
module load rocm/7.2 libffi/3.3

# Re-activate venv
source venv/bin/activate

# Add PyTorch lib directory from venv to LD_LIBRARY_PATH
# This ensures caffe2 libraries are found from the venv installation
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${ROCM_PATH}/lib:$LD_LIBRARY_PATH

# Optional: Add to ~/.bashrc for persistence
# echo "export LD_LIBRARY_PATH=\$(python3 -c \"import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\"):\${ROCM_PATH}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install DeepSpeed
pip3 install deepspeed

# Verify DeepSpeed installation
python3 -c "from deepspeed.profiling.flops_profiler import FlopsProfiler; print('DeepSpeed installed successfully.')"

# Install additional dependencies (if needed)
pip3 install -r setup/requirements.txt

# Install rocprof-compute development dependencies (for rocprof-compute profiling)
pip3 install -r setup/requirements_rocprof-compute-develop.txt
```

**Note**: Activate the virtual environment (`source venv/bin/activate`) each time you start a new session.

### Environment Setup — ROCm 7.13 (TheRock nightly)

The setup above (ROCm 7.2 module + PyTorch rocm7.1 nightly) remains fully supported. Use the
instructions below **instead** if you want the newer ROCm 7.13 stack — PyTorch **2.11** and
Triton **3.6** — for gfx942 (MI300X) / gfx950 (MI355X).

ROCm 7.13 is **alpha-only** and is not available as a system module or from the pytorch.org
nightly index. It comes from AMD's [TheRock](https://github.com/ROCm/TheRock) multi-arch pip
channel, which ships a self-contained ROCm runtime as a wheel dependency of `torch` — no
`/opt/rocm` and no `rocm/7.x` module are needed for training.

```bash
# ROCm 7.13 requires Python 3.14 from the module system.
# (The module ships libpython3.14.so.1.0, which pyenv/system Python 3.14 lacks.)
module load python/3.14

# Navigate to TinyOpenFold and create a dedicated venv
cd HPCTrainingExamples/MLExamples/TinyOpenFold
python3 -m venv venv713
source venv713/bin/activate
pip3 install --upgrade pip

# Install PyTorch 2.11 + Triton 3.6 for ROCm 7.13 from TheRock's multi-arch index.
# torch pulls the matching rocm[libraries] wheels in automatically.
pip3 install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    torch torchvision torchaudio triton

# Verify the ROCm 7.13 stack and GPU
python3 -c "import torch, triton; print('torch', torch.__version__); print('triton', triton.__version__); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
# Expected:
#   torch 2.11.0+rocm7.13.0aYYYYMMDD
#   triton 3.6.0+rocm7.13.0aYYYYMMDD
#   GPU AMD Instinct MI300X

# Install the remaining Python dependencies (DeepSpeed, plotting, etc.)
pip3 install deepspeed
pip3 install -r setup/requirements.txt
```

Notes for the ROCm 7.13 path:
- **No `LD_LIBRARY_PATH` fix is needed.** The `libcaffe2_nvrtc.so` workaround only applies to the
  ROCm 7.2 / pytorch-nightly install above; the TheRock wheels bundle their own runtime.
- **ROCm hardware profilers** (`rocprofv3`, `rocprof-compute`, `rocprof-sys`) used by V2/V3 are
  *not* included in the training wheels. For those, install the full TheRock SDK with the `devel`
  extra (`rocm[libraries,devel,device-gfx942]==7.13.*`) and source its environment — see
  `~/software/therock_install/` for a working install and `sourceme` script.
- Re-run `module load python/3.14` and `source venv713/bin/activate` each new session.

### Basic Training

```bash
# Run with default configuration (64 residues, 16 MSA sequences)
python3 tiny_openfold_v1.py --batch-size 4 --seq-len 64 --num-steps 30

# Expected output:
# Total parameters: ~2.6M
# Model size: ~10.6 MB (FP32)
# Training speed: varies by hardware
```

### With Profiling

```bash
# Enable PyTorch profiler
python3 tiny_openfold_v1.py --enable-pytorch-profiler --profile-dir ./profiles

# View results in TensorBoard
tensorboard --logdir ./profiles
```

### Advanced Configuration

```bash
# Larger model
python3 tiny_openfold_v1.py \
    --msa-dim 128 \
    --pair-dim 256 \
    --num-blocks 8 \
    --seq-len 128 \
    --batch-size 2

# With memory profiling
python3 tiny_openfold_v1.py \
    --enable-all-profiling \
    --profile-dir ./complete_analysis

# Mixed precision training
python3 tiny_openfold_v1.py --use-amp --batch-size 8
```

### Multi-GPU Training

TinyOpenFold supports multi-GPU training using PyTorch's `nn.DataParallel`:

```bash
# Single GPU (explicit)
python3 tiny_openfold_v1.py --device 0 --batch-size 8

# Multi-GPU via environment variables (automatic)
# ROCm (AMD GPUs)
ROCR_VISIBLE_DEVICES=0,1,2,3 python3 tiny_openfold_v1.py --batch-size 32

# CUDA (NVIDIA GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tiny_openfold_v1.py --batch-size 32

# Disable DataParallel even with multiple GPUs visible
python3 tiny_openfold_v1.py --no-data-parallel --device 0
```

**Best Practice:** Scale batch size proportionally with GPU count (e.g., 8 samples per GPU).

### Scaling Studies

Run multi-GPU scaling experiments to measure performance:

```bash
cd version1_pytorch_baseline

# Quick scaling test (1, 2, 4, 8 GPUs)
chmod +x quick_scaling_test.sh
./quick_scaling_test.sh

# Comprehensive scaling study with custom options
chmod +x run.sh
./run.sh --gpus "1 2 4 8" --batch-per-gpu 8 --steps 100

# With mixed precision
./run.sh --amp --steps 50

# Multiple runs for statistics
./run.sh --runs 3 --output-dir scaling_analysis
```

**Example Output:**
```
GPUs  Throughput (s/s)  Speedup  Efficiency
----  ----------------  -------  ----------
1     166.9             1.00x    100.0%
2     202.7             1.21x     60.5%
4     245.3             1.47x     36.8%
8     249.1             1.49x     18.6%
```

See [`version1_pytorch_baseline/README.md`](version1_pytorch_baseline/README.md) for detailed multi-GPU documentation.

## Architecture Overview

### The Evoformer

The Evoformer is the heart of AlphaFold 2, processing two coupled representations:

1. **MSA Representation** `(N_seqs × N_res × msa_dim)`
   - Features for each residue in each sequence of the MSA
   - Updated via row-wise and column-wise attention

2. **Pair Representation** `(N_res × N_res × pair_dim)`
   - Pairwise features between all residues
   - Updated via triangle operations and attention

### Key Components

#### MSA Processing
- **Row-wise Attention**: Attention across residues within each MSA sequence, biased by pair representation
- **Column-wise Attention**: Communication between different sequences at each position
- **MSA Transition**: Point-wise feed-forward network

#### Pair Processing
- **Outer Product Mean**: Projects MSA patterns onto pairwise space
- **Triangle Multiplicative Updates**: Geometric reasoning (if i-j and j-k are close, i-k should be considered)
- **Triangle Self-Attention**: Attention over edges in the residue graph
- **Pair Transition**: Point-wise feed-forward network

#### Structure Module
- Simplified distance prediction from pair representation
- In full AlphaFold 2, this is the Invariant Point Attention (IPA) module

### Parameter Count

**Default Configuration (TinyOpenFoldConfig)**:
- MSA dim: 64, Pair dim: 128
- Evoformer blocks: 4
- Total parameters: **~2.64M**
- Model size: **~10.6 MB (FP32)**, **~5.3 MB (FP16)**

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed parameter calculations.

## Directory Structure

```
TinyOpenFold/
├── README.md                              # This file
├── ARCHITECTURE.md                        # Detailed architecture documentation
└── version1_pytorch_baseline/
    ├── tiny_openfold_v1.py               # Main implementation
    └── README.md                          # Version-specific guide
```

## Performance Characteristics

### Computational Complexity

The Evoformer has interesting scaling properties:

- **MSA Row Attention**: O(N_seqs × N_res² × msa_dim)
- **MSA Column Attention**: O(N_res × N_seqs² × msa_dim)
- **Triangle Operations**: O(N_res³ × pair_dim) ⚠️ Most expensive!
- **Outer Product**: O(N_seqs × N_res² × outer_dim²)

For typical configurations (N_res=64-256):
- Triangle operations dominate computational cost
- Memory usage grows quadratically with sequence length (pair representation)
- MSA depth affects column attention cost

### Typical Performance

*Hardware: AMD MI250X / NVIDIA A100*

| Config | Seq Len | MSA Seqs | Params | Memory | Speed |
|--------|---------|----------|--------|--------|-------|
| Small | 64 | 16 | 2.6M | ~100 MB | ~8-10 samples/sec |
| Medium | 128 | 32 | 10.5M | ~400 MB | ~2-3 samples/sec |
| Large | 256 | 64 | 42M | ~1.6 GB | ~0.5-1 samples/sec |

*Note: Performance varies significantly by hardware and configuration*

## Educational Use Cases

### 1. Understanding AlphaFold 2

Study how the key innovations work:
- Examine `EvoformerBlock` to see how MSA and pair representations interact
- Explore `TriangleMultiplication` to understand geometric reasoning
- Analyze `MSARowAttentionWithPairBias` to see how pair info guides MSA attention

### 2. Profiling and Optimization

Use this as a baseline for optimization experiments:
- Profile with PyTorch Profiler to identify bottlenecks
- Experiment with different attention implementations
- Test kernel fusion opportunities
- Compare with production implementations

### 3. Research and Experimentation

Modify the architecture to test ideas:
- Change attention patterns
- Experiment with different update mechanisms
- Test alternative structure modules
- Implement custom operators

## Differences from Production AlphaFold 2

This is an **educational simplification**. Key differences:

| Aspect | TinyOpenFold | AlphaFold 2 |
|--------|--------------|-------------|
| Evoformer blocks | 4 | 48 |
| Dimensions | 64/128 | 256/128 |
| Templates | ❌ None | ✅ Template featurization |
| Structure Module | Simple distance prediction | Full IPA with frames |
| Recycling | ❌ Single pass | ✅ Multiple iterations |
| Data | Synthetic | Real MSAs and structures |
| Purpose | Education/Profiling | Production prediction |

## Command Line Options

```bash
# Model Configuration
--msa-dim 64              # MSA representation dimension
--pair-dim 128            # Pair representation dimension
--num-blocks 4            # Number of Evoformer blocks
--num-seqs 16             # Number of MSA sequences
--seq-len 64              # Sequence length (number of residues)

# Training Configuration
--num-steps 50            # Training iterations
--batch-size 4            # Batch size
--learning-rate 3e-4      # Learning rate
--use-amp                 # Enable mixed precision

# Profiling Options
--enable-pytorch-profiler # Enable PyTorch profiler
--enable-memory-profiling # Track memory usage
--enable-all-profiling    # Enable all profiling features
--profile-dir ./profiles  # Output directory for profiles
--warmup-steps 3          # Profiler warmup steps
--profile-steps 5         # Steps to profile

# Utilities
--validate-setup          # Run validation checks
```

## Understanding the Output

During training, you'll see:

```
Model Configuration:
   MSA dimension: 64
   Pair dimension: 128
   Evoformer blocks: 4
   Total parameters: 2,641,728
   Model size: 10.6 MB (FP32)

Training Configuration:
   Training steps: 50
   Batch size: 4
   Device: CUDA

Step   0/50 | Loss: 45.2341 | Speed:   8.5 samples/sec | Memory:  102.3 MB | Time:  470.2ms
Step  10/50 | Loss: 38.7123 | Speed:   9.1 samples/sec | Memory:  102.3 MB | Time:  439.5ms
```

**Key Metrics**:
- **Loss**: MSE on predicted distances (should decrease over time)
- **Speed**: Samples processed per second
- **Memory**: GPU memory allocated
- **Time**: Time per training step

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

```bash
# Reduce batch size
python3 tiny_openfold_v1.py --batch-size 2

# Reduce sequence length
python3 tiny_openfold_v1.py --seq-len 32

# Reduce MSA sequences
python3 tiny_openfold_v1.py --num-seqs 8

# Use mixed precision
python3 tiny_openfold_v1.py --use-amp
```

### Slow Performance

The triangle operations are O(N³) and can be slow:

```bash
# Use smaller sequences
python3 tiny_openfold_v1.py --seq-len 32

# Reduce Evoformer blocks
python3 tiny_openfold_v1.py --num-blocks 2

# Profile to identify bottlenecks
python3 tiny_openfold_v1.py --enable-pytorch-profiler
```

## Further Reading

### AlphaFold 2 Resources

- **Paper**: [Jumper et al., "Highly accurate protein structure prediction with AlphaFold", Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)
- **Supplement**: Detailed architectural descriptions
- **OpenFold**: https://github.com/aqlaboratory/openfold - Full production implementation
- **AlphaFold GitHub**: https://github.com/deepmind/alphafold - Original DeepMind code

### Understanding the Evoformer

- AlphaFold 2 Supplement, Section 1.6: Evoformer architecture
- Section 1.6.7-1.6.8: Triangle multiplicative updates
- Section 1.7: Outer product mean
- Section 1.8: Structure module and IPA

### Related Topics

- **Attention Mechanisms**: Understanding multi-head attention
- **Geometric Deep Learning**: Graph neural networks for 3D structures
- **Protein Structure Prediction**: MSAs, templates, and structural biology

## Contributing

This is an educational project. Improvements welcome:

- Enhanced documentation
- Additional visualization tools
- Performance optimizations
- Extended architecture variants

## Citation

If you use TinyOpenFold in your work, please cite both this implementation and the original AlphaFold 2:

```bibtex
@article{jumper2021alphafold,
  title={Highly accurate protein structure prediction with AlphaFold},
  author={Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  journal={Nature},
  volume={596},
  number={7873},
  pages={583--589},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## License

Apache 2.0 License - See LICENSE file for details

## Acknowledgments

- Based on AlphaFold 2 by DeepMind
- Inspired by OpenFold (https://github.com/aqlaboratory/openfold)
- Educational structure follows TinyLLaMA example

---

**Ready to explore AlphaFold 2? Start with:**

```bash
cd version1_pytorch_baseline
python3 tiny_openfold_v1.py --validate-setup
```

