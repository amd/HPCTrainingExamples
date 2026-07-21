# ROCm PyTorch Inference Benchmark Workshop
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

## Table of Contents

1. [Introduction & Setup](#1-introduction--setup)
2. [Understanding the Benchmark Tool](#2-understanding-the-benchmark-tool)
3. [Exercise 1: Single GPU Baseline](#3-exercise-1-single-gpu-baseline)
4. [Exercise 2: Precision Comparison](#4-exercise-2-precision-comparison-fp32-vs-fp16)
5. [Exercise 3: PyTorch Profiler Integration](#5-exercise-3-pytorch-profiler-integration)
6. [Exercise 4: DeepSpeed FLOPS Profiler](#6-exercise-4-deepspeed-flops-profiler)
7. [Exercise 5: Multi-GPU Scaling](#7-exercise-5-multi-gpu-scaling)
8. [Exercise 6: PyTorch 2.0 Compilation](#8-exercise-6-pytorch-20-compilation)
9. [Exercise 7: ROCm Profiler Integration](#9-exercise-7-rocm-profiler-integration)
10. [Wrap-up & Best Practices](#10-wrap-up--best-practices)

---

## 1. Introduction & Setup

### 1.1 What is Inference?

**Inference** is the process of using a trained neural network to make predictions on new data.

**Key Differences from Training:**

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Purpose** | Learn patterns from data | Make predictions |
| **Direction** | Forward + Backward pass | Forward pass only |
| **Gradients** | Required | Not required |
| **Batch Size** | Usually larger | Often smaller (1-32) |
| **Performance Goal** | Throughput (samples/sec) | Latency (ms/sample) AND throughput |
| **Memory Usage** | High (stores activations) | Lower (no gradient storage) |

**Why Benchmark Inference?**

- Optimize for production deployment
- Understand hardware utilization
- Compare different models
- Justify hardware purchases
- Identify bottlenecks

### 1.2 Workshop Goals

By the end of this workshop, you will:

- Run standardized inference benchmarks on AMD GPUs
- Use PyTorch Profiler to identify bottlenecks
- Understand FLOPS efficiency with DeepSpeed profiler
- Scale workloads across multiple GPUs
- Apply PyTorch 2.0 compilation optimizations
- Use ROCm profiling tools for kernel-level analysis
- Interpret performance metrics and make optimization decisions

### 1.3 Environment Verification

Let's verify your system is ready for the workshop.

#### Step 1: Check ROCm Installation

```bash
# Check if ROCm is installed
rocminfo | grep "Name:"
```

**Expected Output:**
```
  Name:                    gfx942
  Name:                    AMD Instinct MI325X
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
GPU[0]    : GPU Name: AMD Instinct MI325X
GPU[0]    : Temperature: 35.0°C
GPU[0]    : GPU Memory Usage: 256 MB / 196608 MB
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

**Error: "Permission denied"**
```bash
# Check if you're in the right groups
groups | grep video
groups | grep render

# If not, add yourself (requires sudo)
sudo usermod -aG video $USER
sudo usermod -aG render $USER
# Logout/login required!
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
GPU Name: AMD Instinct MI325X
GPU Memory: 196.6 GB
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

#### Step 4: Verify Benchmark Script

```bash
# Navigate to inference benchmark directory
cd inference_benchmark/

# List files
ls -la
```

**Expected Output:**
```
-rw-rw-r-- micro_benchmarking_pytorch.py
-rw-rw-r-- README.md
-rw-rw-r-- fp16util.py
drwxrwxr-x TorchTensorOpsBench/
```



#### Step 5: Quick Test Run

Let's verify everything works with a very small test:

```bash
# Run a tiny test (should complete in ~30 seconds)
python3 micro_benchmarking_pytorch.py --network resnet18 --batch-size 16 --iterations 5
```

**Expected Output:**
```
Using network: resnet18
Batch size: 16
Iterations: 5
FP16: False

Epoch 0: Loss = 6.9088, Time = 0.125 seconds
Epoch 1: Loss = 6.9088, Time = 0.042 seconds
Epoch 2: Loss = 6.9088, Time = 0.041 seconds
Epoch 3: Loss = 6.9088, Time = 0.041 seconds
Epoch 4: Loss = 6.9088, Time = 0.040 seconds

Average time per iteration: 0.041 seconds
Throughput: 390.2 samples/sec
```

**If you see this output, your environment is ready!**

### 1.4 Understanding Key Metrics

Before we begin the exercises, let's understand what we're measuring:

#### Throughput (samples/sec or images/sec)
- **What:** Number of samples processed per second
- **Higher is better**
- **Use case:** Batch inference, data center deployments
- **Formula:** `(batch_size × num_iterations) / total_time`

#### Latency (milliseconds)
- **What:** Time to process a single sample or batch
- **Lower is better**
- **Use case:** Real-time applications, interactive systems
- **Formula:** `total_time / num_iterations`

#### Memory Usage (MB or GB)
- **What:** GPU memory consumed by model and data
- **Lower is better (allows larger batches)**
- **Includes:** Model weights, activations, gradients (if training)

#### GPU Utilization (%)
- **What:** Percentage of GPU compute used
- **Higher is better (approaching 100%)**
- **Note:** Can be low if memory-bound or CPU-bound

#### FLOPS (Floating Point Operations Per Second)
- **What:** Computational throughput
- **Higher is better**
- **Theoretical vs Achieved:** Gap indicates optimization opportunity

---

## 2. Understanding the Benchmark Tool

### 2.1 What is `micro_benchmarking_pytorch.py`?

This is a standardized tool for benchmarking PyTorch inference on ROCm.

**Purpose:**
- Measure inference performance across different models
- Compare hardware configurations
- Test optimization techniques
- Standardized, reproducible results

**Features:**
- 50+ pre-configured models (ResNet, VGG, EfficientNet, ViT, etc.)
- FP32 and FP16 precision support
- Single and multi-GPU support
- PyTorch Profiler integration
- DeepSpeed FLOPS profiler integration
- PyTorch 2.0 compilation support

### 2.2 Available Models

The benchmark includes many popular vision models:

**Classification Models:**
```python
# ResNet family (most commonly used for benchmarking)
resnet18, resnet34, resnet50, resnet101, resnet152

# EfficientNet family (efficient models)
efficientnet_b0, efficientnet_b1, ..., efficientnet_b7

# Vision Transformers (attention-based)
vit_b_16, vit_b_32, vit_l_16, vit_h_14

# MobileNet (mobile/edge optimized)
mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

# VGG (classic architecture)
vgg11, vgg13, vgg16, vgg19

# And many more...
```

**Segmentation Models:**
```python
fcn_resnet50, fcn_resnet101
deeplabv3_resnet50, deeplabv3_resnet101
```

**For this workshop, we'll focus on ResNet50 because:**
- Industry-standard benchmark
- Good balance of compute and memory operations
- Well-optimized by hardware vendors
- Comparable results across papers and benchmarks

### 2.3 Command-Line Arguments

Let's understand the key arguments:

#### Basic Arguments

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50        # Model to benchmark
    --batch-size 64           # Number of samples per batch
    --iterations 20           # Number of iterations to run
```

#### Precision Arguments

```bash
--fp16 1                      # Use FP16 (half precision)
--amp-opt-level 2             # Use automatic mixed precision (APEX)
```

#### Profiling Arguments

```bash
--autograd-profiler           # Enable PyTorch autograd profiler
--kineto                      # Enable Kineto profiler (PyTorch 1.8+)
--flops-prof-step 10          # Enable DeepSpeed FLOPS profiler at step 10
```

#### Multi-GPU Arguments

```bash
# Option 1: Using torchrun (recommended)
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50

# Option 2: Manual distributed setup
--distributed_dataparallel    # Enable distributed data parallel
--device_ids 0,1              # GPUs to use
--rank 0                      # Process rank
--world-size 2                # Total number of processes
```

#### PyTorch 2.0 Arguments

```bash
--compile                     # Enable torch.compile
--compileContext "{'mode': 'max-autotune'}"  # Compilation options
```

### 2.4 Understanding Output

When you run the benchmark, you'll see output like this:

```
Using network: resnet50
Batch size: 64
Iterations: 20
FP16: False

Warming up...
Warmup complete.

Epoch 0: Loss = 6.9088, Time = 0.145 seconds
Epoch 1: Loss = 6.9088, Time = 0.042 seconds
Epoch 2: Loss = 6.9088, Time = 0.041 seconds
...
Epoch 19: Loss = 6.9088, Time = 0.040 seconds

========================================
Performance Summary:
========================================
Average time per iteration: 0.041 seconds
Throughput: 1560.9 samples/sec
Memory usage: 4523 MB
========================================
```

**Let's break this down:**

1. **Configuration Echo**
   - Shows your settings
   - Verify these are correct before trusting results

2. **Warmup Phase**
   - First few iterations are slower (kernel compilation, cache warming)
   - Results are discarded

3. **Timed Iterations**
   - Each iteration shows loss and time
   - Loss should be consistent (model is random, not trained)

4. **Performance Summary**
   - **Average time:** Excludes warmup, arithmetic mean
   - **Throughput:** samples/sec = (batch_size × iterations) / total_time
   - **Memory:** Peak GPU memory usage

### 2.5 Creating Your Results Template

Let's create a file to track your results throughout the workshop:

```bash
# Create results file
cat > my_workshop_results.txt << 'EOF'
================================================================================
ROCm PyTorch Inference Benchmark Workshop Results
================================================================================
Name: [Your Name]
Date: [Today's Date]
GPU: [Your GPU Model from rocm-smi]
ROCm Version: [From rocminfo]
PyTorch Version: [From python -c "import torch; print(torch.__version__)"]
================================================================================

Exercise 1: Single GPU Baseline (ResNet50, FP32, BS=32)
------------------------------------------------------------------------
Throughput:          __________ samples/sec
Memory Usage:        __________ MB
Avg Time/Iteration:  __________ seconds
Notes:


Exercise 2: Precision Comparison (ResNet50, FP16, BS=32)
------------------------------------------------------------------------
FP32 Throughput:     __________ samples/sec
FP16 Throughput:     __________ samples/sec
Speedup (FP16/FP32): __________x
Memory Reduction:    __________%
Notes:


Exercise 3: PyTorch Profiler
------------------------------------------------------------------------
Top 5 Slowest Operations:
1. ____________________: __________ ms
2. ____________________: __________ ms
3. ____________________: __________ ms
4. ____________________: __________ ms
5. ____________________: __________ ms
Notes:


[Continue for remaining exercises...]
EOF
```

**Open this file in a text editor and fill it out as you complete each exercise!**

---

## 3. Exercise 1: Single GPU Baseline

### 3.1 Objective

Run your first benchmark and establish a baseline for comparison.

**What you'll learn:**
- How to run the benchmark tool
- How to interpret basic output
- What "good" performance looks like
- How to verify your results

### 3.2 Step-by-Step Instructions

#### Step 1: Navigate to the benchmark directory

```bash
cd ~/castille-ai-workshop-training/inference_benchmark/
```

#### Step 2: Run the baseline benchmark

```bash
# Run ResNet50 with batch size 32 for 20 iterations
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 20
```


#### Step 3: Watch the output

You'll see output like this:

```
Using network: resnet50
Batch size: 32
Iterations: 20
FP16: False
Device: cuda:0

Loading model...
Model loaded successfully.

Warming up (5 iterations)...
Warmup iteration 0: Loss = 6.9078, Time = 0.242 seconds
Warmup iteration 1: Loss = 6.9078, Time = 0.065 seconds
Warmup iteration 2: Loss = 6.9078, Time = 0.064 seconds
Warmup iteration 3: Loss = 6.9078, Time = 0.063 seconds
Warmup iteration 4: Loss = 6.9078, Time = 0.063 seconds
Warmup complete.

Running timed iterations...
Epoch 0: Loss = 6.9078, Time = 0.063 seconds
Epoch 1: Loss = 6.9078, Time = 0.062 seconds
Epoch 2: Loss = 6.9078, Time = 0.062 seconds
...
Epoch 19: Loss = 6.9078, Time = 0.062 seconds

========================================
Performance Summary:
========================================
Network: resnet50
Batch size: 32
Iterations: 20 (excluding warmup)
Precision: FP32

Average time per iteration: 0.062 seconds
Standard deviation: 0.001 seconds
Throughput: 516.1 samples/sec
GPU Memory Usage: 4523 MB

Images per second: 516.1
Milliseconds per batch: 62.0
Microseconds per sample: 1937.5
========================================
```

### 3.3 Understanding Your Results

Let's analyze what these numbers mean:

#### 1. Warmup Phase
```
Warmup iteration 0: Loss = 6.9078, Time = 0.242 seconds  ← SLOW (first run)
Warmup iteration 1: Loss = 6.9078, Time = 0.065 seconds  ← Much faster
Warmup iteration 2: Loss = 6.9078, Time = 0.064 seconds  ← Stable
```

**Why is the first iteration slow?**
- Kernel compilation (Triton, ROCm)
- GPU memory allocation
- Cache warming
- cuDNN/MIOpen autotuning

**This is normal! Always exclude warmup from measurements.**

#### 2. Throughput: 516.1 samples/sec

**What does this mean?**
- Your GPU can process 516 images per second
- For batch size 32: 516.1 / 32 = 16.1 batches/second

**Is this good?**
- For ResNet50 FP32 on MI200 series: 450-550 samples/sec is typical
- For MI300 series: 600-800 samples/sec is typical
- For older GPUs (V100, MI100): 300-400 samples/sec is typical

#### 3. Memory Usage: 4523 MB

**What uses this memory?**
- Model weights: ~100 MB (ResNet50 has 25.6M parameters × 4 bytes)
- Input batch: 32 × 3 × 224 × 224 × 4 bytes = ~19 MB
- Activations: ~4400 MB (intermediate feature maps)

**Why so much for activations?**
- ResNet50 has many layers (50!)
- Each layer creates feature maps
- Feature maps are large (early layers: 32 × 64 × 112 × 112 × 4 bytes = 102 MB EACH!)

#### 4. Time Consistency
```
Standard deviation: 0.001 seconds
```

**This is important!**
- Low std dev (< 5% of mean): Stable, trustworthy results
- High std dev (> 10% of mean): Something is wrong (thermal throttling, system interference)

### 3.4 Checkpoint: Verify Your Results

Before moving on, check:

- [ ] Throughput is between 300-800 samples/sec (depending on GPU)
- [ ] Memory usage is around 4000-5000 MB
- [ ] Standard deviation is small (< 0.005 seconds)
- [ ] All iterations show same loss (~6.9)
- [ ] No error messages

**If all checks pass, record your results and continue!**

**If something looks wrong:**

**Problem:** Throughput very low (< 100 samples/sec)
```bash
# Check GPU utilization
rocm-smi

# Should show ~100% during benchmark
# If low, check:
# 1. CPU bottleneck (increase --batch-size)
# 2. Slow storage (model loading)
# 3. System interference (close other programs)
```

**Problem:** Memory usage extremely high (> 10000 MB)
```bash
# Reduce batch size
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 16 --iterations 20
```

**Problem:** Inconsistent results (high std dev)
```bash
# Increase iterations for better averaging
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 50
```

### 3.5 Recording Your Results

Record these values in your `my_workshop_results.txt`:

```
Exercise 1: Single GPU Baseline (ResNet50, FP32, BS=32)
------------------------------------------------------------------------
Throughput:          516.1 samples/sec
Memory Usage:        4523 MB
Avg Time/Iteration:  0.062 seconds
GPU Model:           AMD Instinct MI325X
Notes:
- Warmup took 5 iterations
- Results very stable (std dev 0.001s)
- Baseline for all future comparisons
```

### 3.6 Optional: Try Different Batch Sizes

**Why does batch size matter?**

Larger batches improve GPU utilization but increase memory usage.

```bash
# Small batch
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 8 --iterations 20

# Medium batch (your baseline)
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 20

# Large batch
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 20

# Very large batch (might OOM!)
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 20
```

**Create a quick comparison table:**

| Batch Size | Throughput (samples/sec) | Memory (MB) | Samples/sec per GB |
|------------|-------------------------|-------------|-------------------|
| 8          | ?                       | ?           | ?                 |
| 32         | 516.1                   | 4523        | 0.114             |
| 128        | ?                       | ?           | ?                 |
| 256        | OOM or ?                | ?           | ?                 |

**What do you observe?**
- Throughput increases with batch size... but not linearly
- Memory increases with batch size
- There's a sweet spot for efficiency

---

## 4. Exercise 2: Precision Comparison (FP32 vs FP16)

### 4.1 Objective

Compare FP32 (32-bit floating point) vs FP16 (16-bit floating point) precision.

**What you'll learn:**
- What FP16 is and why it matters
- Performance benefits of reduced precision
- Memory savings from FP16
- When to use FP16 vs FP32

### 4.2 What is FP16?

**Floating Point Precision:**

```
FP32 (Float32):  32 bits = 1 sign + 8 exponent + 23 mantissa
                 Range: ±1.4 × 10⁻⁴⁵ to ±3.4 × 10³⁸
                 Precision: ~7 decimal digits

FP16 (Float16):  16 bits = 1 sign + 5 exponent + 10 mantissa
                 Range: ±6.0 × 10⁻⁸ to ±6.5 × 10⁴
                 Precision: ~3 decimal digits
```

**Benefits of FP16:**
- 2x less memory (16 bits vs 32 bits)
- 2x more data per memory transaction
- 2-4x faster compute (specialized hardware)
- Lower power consumption

**Drawbacks of FP16:**
- Lower precision (can cause numerical issues)
- Smaller range (risk of overflow/underflow)
- Requires careful model design

**For inference:** FP16 is usually safe and recommended!

### 4.3 Running FP32 Baseline (Repeat)

First, let's re-run FP32 to have a fresh comparison:

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 20 \
    --fp16 0
```

**Record the results:**
```
FP32 Throughput: __________ samples/sec
FP32 Memory:     __________ MB
```

### 4.4 Running FP16 Benchmark

Now let's run with FP16:

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 20 \
    --fp16 1
```

**Expected output:**
```
Using network: resnet50
Batch size: 32
Iterations: 20
FP16: True  ← Notice this!
Device: cuda:0

Converting model to FP16...
Model conversion complete.

Warming up...
Warmup complete.

Running timed iterations...
Epoch 0: Loss = 6.9062, Time = 0.031 seconds  ← MUCH FASTER!
Epoch 1: Loss = 6.9062, Time = 0.030 seconds
...

========================================
Performance Summary:
========================================
Network: resnet50
Batch size: 32
Precision: FP16  ← Notice this!

Average time per iteration: 0.031 seconds
Throughput: 1032.3 samples/sec  ← ~2x faster!
GPU Memory Usage: 2834 MB        ← ~37% less memory!
========================================
```

### 4.5 Analyzing the Results

Let's compare FP32 vs FP16:

#### Create a comparison table:

```
┌──────────────────────┬───────────┬───────────┬──────────────┐
│ Metric               │ FP32      │ FP16      │ Improvement  │
├──────────────────────┼───────────┼───────────┼──────────────┤
│ Throughput (samp/s)  │ 516.1     │ 1032.3    │ 2.00x faster │
│ Memory (MB)          │ 4523      │ 2834      │ 37% less     │
│ Time per batch (ms)  │ 62.0      │ 31.0      │ 2.00x faster │
│ Numerical accuracy   │ Full      │ Reduced   │ -            │
└──────────────────────┴───────────┴───────────┴──────────────┘
```

#### Why is it faster?

1. **Less Memory Traffic:**
   - FP16 tensor: half the size
   - Loading weights from memory: 2x faster
   - Writing activations: 2x faster

2. **Specialized Hardware:**
   - AMD MI200/MI300: Matrix Core FP16 instructions
   - 2-4x higher TFLOPS for FP16 vs FP32

3. **Cache Efficiency:**
   - More data fits in L2 cache
   - Fewer cache misses

#### Why less memory?

```
Model weights:  25.6M params × 2 bytes = 51 MB (vs 102 MB in FP32)
Activations:    ~2200 MB (vs ~4400 MB in FP32)
Input batch:    32 × 3 × 224 × 224 × 2 bytes = ~9.6 MB (vs ~19 MB)
```

### 4.6 When to Use FP16?

**Use FP16 when:**
- Inference only (no gradient accumulation issues)
- Large models (memory constrained)
- Throughput matters more than last-bit accuracy
- Model is not numerically sensitive

**Avoid FP16 when:**
- Need exact numerical reproducibility
- Model has numerical instability
- Small model (no memory benefit)
- Training (use mixed precision instead)

### 4.7 Testing Numerical Accuracy

Let's verify FP16 doesn't hurt model accuracy significantly.

#### Run both and compare loss:

```bash
# FP32
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 5 --fp16 0 | grep "Epoch 4"

# FP16
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 5 --fp16 1 | grep "Epoch 4"
```

**Expected output:**
```
FP32: Epoch 4: Loss = 6.9078
FP16: Epoch 4: Loss = 6.9062
```

**Difference:** 0.0016 (0.02%)

**This is negligible!**

### 4.8 Checkpoint

Before continuing:

- [ ] FP16 is ~2x faster than FP32
- [ ] FP16 uses ~30-40% less memory
- [ ] Loss values are very similar (~0.02% difference)
- [ ] You understand when to use FP16

**Record your results in `my_workshop_results.txt`!**

### 4.9 Advanced: Maximum Batch Size

Let's find the maximum batch size for both precisions:

```bash
# FP32 - keep increasing until OOM
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 5 --fp16 0
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 5 --fp16 0
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 5 --fp16 0

# FP16 - should go much higher!
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 5 --fp16 1
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 5 --fp16 1
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 512 --iterations 5 --fp16 1
```

**Track maximum batch sizes:**
```
FP32 max batch size: __________ (before OOM)
FP16 max batch size: __________ (before OOM)

Ratio: FP16 supports __________x larger batches!
```

---

## 5. Exercise 3: PyTorch Profiler Integration

### 5.1 Objective

Use PyTorch's built-in profiler to identify performance bottlenecks.

**What you'll learn:**
- How to enable PyTorch Profiler
- Reading profiler output
- Identifying slow operations
- Understanding CPU vs GPU time

### 5.2 What is PyTorch Profiler?

**PyTorch Profiler** provides detailed performance analysis:

- **Operator-level timing:** How long each operation takes
- **CPU vs GPU time:** Distinguish CPU overhead from GPU compute
- **Memory profiling:** Track memory allocations
- **Stack traces:** See which code triggered operations
- **Kernel details:** See GPU kernel launches

**When to use:**
- Identifying bottleneck operations
- Finding CPU overhead
- Optimizing custom operations
- Debugging slow models

### 5.3 Running with PyTorch Profiler

Let's modify our benchmark to use the profiler.

#### Step 1: Run with profiler enabled

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 10 \
    --fp16 0 \
    --autograd-profiler
```



#### Step 2: Understanding the output

You'll see LOTS of output! Let's focus on key sections:

```
========================================
PyTorch Profiler Results:
========================================

Top 10 operations by total CPU time:
---------------------------  ------------  ------------  ------------
Name                         Self CPU %    Self CPU      CPU total
---------------------------  ------------  ------------  ------------
aten::convolution            5.23%         128.45ms      8.52s
aten::batch_norm             2.15%         52.75ms       1.32s
aten::relu_                  1.87%         45.91ms       45.91ms
aten::max_pool2d             0.95%         23.32ms       67.45ms
aten::addmm                  0.78%         19.15ms       234.67ms
aten::linear                 0.65%         15.95ms       250.62ms
aten::add_                   0.52%         12.78ms       12.78ms
aten::_convolution           4.87%         119.55ms      8.40s
aten::cudnn_convolution      78.23%        1.92s         1.92s
...
---------------------------  ------------  ------------  ------------

Top 10 operations by total CUDA time:
---------------------------  ------------  ------------  ------------
Name                         Self CUDA     CUDA total    # of Calls
---------------------------  ------------  ------------  ------------
void cudnn::detail::implicit_convolve_sgemm...   1.82s    1.82s      320
void cudnn::bn_fw_tr_1C11...                     234.56ms 234.56ms   160
Memcpy HtoD (Pageable -> Device)                 145.32ms 145.32ms   50
void at::native::vectorized_elementwise...       89.45ms  89.45ms    640
void cudnn::ops::nchwToNhwc...                   67.23ms  67.23ms    160
...
---------------------------  ------------  ------------  ------------

Memory Profiling:
---------------------------  ------------  ------------  ------------
Name                         CPU Mem       CUDA Mem      # of Calls
---------------------------  ------------  ------------  ------------
aten::convolution            0 b           3.52 Gb       320
aten::batch_norm             0 b           834.56 Mb     160
aten::relu_                  0 b           0 b           160
aten::max_pool2d             0 b           256.00 Mb     32
...
---------------------------  ------------  ------------  ------------
```

### 5.4 Interpreting the Results

#### 1. CPU Time vs CUDA Time

**CPU Time:** Time spent on Python/CPU side
- Launching kernels
- Python overhead
- Data preparation

**CUDA Time:** Time spent on GPU
- Actual computation
- Memory transfers
- Kernel execution

**Key insight:** If CPU time >> CUDA time, you have CPU overhead!

#### 2. Top Operations

From the example above:

```
Top operation: cudnn_convolution (78.23% of CPU time)
```

**What this means:**
- Convolutions dominate runtime
- This is expected for ResNet50!
- Optimizing convolutions = biggest impact

#### 3. Memory Allocation

```
aten::convolution: 3.52 GB CUDA memory
```

**What this means:**
- Convolutions use most memory
- Intermediate feature maps are large
- This is why batch size is limited

### 5.5 Hands-On: Finding Bottlenecks

Let's analyze YOUR profiler output:

#### Task 1: Find the top 5 slowest operations

Look at "Top 10 operations by total CUDA time" and write down:

```
1. ___________________________: ___________ ms
2. ___________________________: ___________ ms
3. ___________________________: ___________ ms
4. ___________________________: ___________ ms
5. ___________________________: ___________ ms
```

#### Task 2: Calculate convolution percentage

```
Total CUDA time: ___________ seconds
Convolution CUDA time: ___________ seconds
Percentage: (___________ / ___________) × 100 = _________%
```

**Is convolution the bottleneck?**
- If > 70%: Yes, convolution is the main bottleneck
- If < 50%: Other operations are significant

#### Task 3: Check for CPU overhead

```
Total CPU time: ___________ seconds
Total CUDA time: ___________ seconds
Ratio: ___________ / ___________ = ___________
```

**Interpretation:**
- Ratio < 1.2: Good! Low CPU overhead
- Ratio 1.2-2.0: Moderate CPU overhead
- Ratio > 2.0: High CPU overhead!

### 5.6 Comparing FP32 vs FP16 Profiling

Let's profile both precisions:

```bash
# FP32
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 10 --fp16 0 --autograd-profiler > profile_fp32.txt

# FP16
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 10 --fp16 1 --autograd-profiler > profile_fp16.txt
```

#### Compare convolution times:

```bash
# FP32 convolution time
grep "cudnn_convolution" profile_fp32.txt | head -1

# FP16 convolution time
grep "cudnn_convolution" profile_fp16.txt | head -1
```

**Create comparison:**
```
FP32 convolution time: ___________ ms
FP16 convolution time: ___________ ms
Speedup: ___________ / ___________ = ___________x
```

### 5.7 Advanced: Chrome Trace Visualization

PyTorch Profiler can export a Chrome trace for visual analysis.

#### Step 1: Create a profiling script

Create a file `profile_resnet.py`:

```python
import torch
import torchvision
import torch.profiler

# Load model
model = torchvision.models.resnet50().cuda()
model.eval()

# Create dummy input
input = torch.randn(32, 3, 224, 224).cuda()

# Warmup
with torch.no_grad():
    for _ in range(5):
        model(input)

# Profile with Chrome trace export
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            model(input)

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export Chrome trace
prof.export_chrome_trace("resnet50_trace.json")
print("\nChrome trace exported to: resnet50_trace.json")
print("View at: chrome://tracing")
```

#### Step 2: Run the script

```bash
python3 profile_resnet.py
```

#### Step 3: View the trace

1. Open Chrome browser
2. Go to `chrome://tracing`
3. Click "Load"
4. Select `resnet50_trace.json`

**You'll see a timeline view:**
- X-axis: Time
- Y-axis: Different operations
- Color: Operation type

**What to look for:**
- Long operations (bottlenecks)
- GPU idle time (gaps)
- Memory transfer time
- Kernel launch overhead

### 5.8 Checkpoint

Before continuing:

- [ ] You can enable PyTorch Profiler with `--autograd-profiler`
- [ ] You can identify top operations by CUDA time
- [ ] You understand CPU time vs CUDA time
- [ ] You can compare FP32 vs FP16 performance at operation level
- [ ] You know how to export Chrome traces for visualization

**Record your top 5 operations in `my_workshop_results.txt`!**

---

## 6. Exercise 4: DeepSpeed FLOPS Profiler

### 6.1 Objective

Measure computational efficiency using DeepSpeed FLOPS Profiler.

**What you'll learn:**
- What FLOPs are and why they matter
- Theoretical vs achieved FLOPS
- Computational efficiency
- Identifying compute vs memory-bound operations

### 6.2 What are FLOPs?

**FLOPS = Floating Point Operations Per Second**

**Key concepts:**

1. **Operation Count:**
   - Total floating-point operations in your model
   - Example: Matrix multiply (M×K) × (K×N) = 2×M×K×N FLOPs

2. **Theoretical Peak:**
   - Maximum FLOPs your hardware can achieve
   - MI325X: ~653 TFLOPS (FP16), ~326 TFLOPS (FP32)

3. **Achieved FLOPs:**
   - What your model actually achieves
   - Usually much lower than peak!

4. **Efficiency:**
   - (Achieved / Theoretical) × 100%
   - 50%+ is very good!
   - 10-20% is typical for many workloads

### 6.3 Why Measure FLOPs?

**FLOPs efficiency tells you:**

- Are you **compute-bound** or **memory-bound**?
  - High efficiency (>40%): Compute-bound (good!)
  - Low efficiency (<20%): Memory-bound (need optimization!)

- How much headroom for optimization?
  - At 10% efficiency: 10x speedup possible!
  - At 80% efficiency: Already well-optimized

- Hardware utilization:
  - Are you getting value from your expensive GPU?

### 6.4 Understanding Compute vs Memory Bound

```
Compute-bound:
- Lots of arithmetic operations
- GPU cores fully utilized
- Examples: Matrix multiply, convolutions with large kernels
- Optimization: Use faster compute (FP16, Tensor Cores)

Memory-bound:
- Lots of memory reads/writes
- Memory bandwidth saturated
- Examples: Element-wise operations, small convolutions, attention
- Optimization: Reduce memory traffic (fusion, better layouts)
```

### 6.5 Running DeepSpeed FLOPS Profiler

#### Step 1: Install DeepSpeed

```bash
# Install DeepSpeed
pip install deepspeed
```

#### Step 2: Run with FLOPS profiler

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 20 \
    --fp16 0 \
    --flops-prof-step 10
```

**Note:** `--flops-prof-step 10` means profile at iteration 10 (after warmup)



#### Step 3: Understanding the output

You'll see extensive output like this:

```
========================================
DeepSpeed FLOPS Profiler Output:
========================================

-------------------------- DeepSpeed Flops Profiler --------------------------

Profile Summary at step 10:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

world size:                        1
data parallel size:                1
model parallel size:               1
batch size per GPU:                32
params per GPU:                    25.56 M
params of model = params per GPU * mp_size:            25.56 M
fwd MACs per GPU:                  4.10 G
fwd FLOPs per GPU:                 8.20 G
fwd FLOPs of model = fwd FLOPs per GPU * mp_size:     8.20 G
fwd latency:                       10.52 ms
bwd latency:                       21.34 ms
fwd FLOPS per GPU = fwd FLOPs per GPU / fwd latency:  779.47 GFLOPS
bwd FLOPS per GPU = 2 * fwd FLOPs per GPU / bwd latency:  768.54 GFLOPS
fwd+bwd FLOPS per GPU = 3 * fwd FLOPs per GPU / (fwd+bwd latency): 772.89 GFLOPS

----------------------------- Aggregated Profile per GPU -----------------------------
Top 10 modules in terms of params, MACs or fwd latency at different model depths:

depth 0:
    params      |    MACs       |  fwd latency  |   module
    25.56 M     |  4.10 G       |  10.52 ms     |  ResNet

depth 1:
    params      |    MACs       |  fwd latency  |   module
    0           |   803.16 M    |   1.23 ms     |  conv1
    0           |   411.04 M    |   1.45 ms     |  layer1
    0           |   822.08 M    |   2.34 ms     |  layer2
    0           |   1.64 G      |   3.67 ms     |  layer3
    0           |   822.08 M    |   1.54 ms     |  layer4
    2.05 M      |   0           |   0.12 ms     |  fc

Top 10 modules in terms of fwd latency:
    fwd latency |   module
    10.52 ms    |  ResNet
    3.67 ms     |  layer3
    2.34 ms     |  layer2
    1.54 ms     |  layer4
    1.45 ms     |  layer1
    1.23 ms     |  conv1
    0.12 ms     |  fc

----------------------------- Detailed Profile per GPU -----------------------------

Each module profile is listed after its name in the following order:
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency

ResNet (25.56 M, 100.00%, 4.10 G, 100.00%, 10.52 ms, 100.00%)
  conv1 (0, 0.00%, 803.16 M, 19.59%, 1.23 ms, 11.69%)
  bn1 (0, 0.00%, 0, 0.00%, 0.34 ms, 3.23%)
  relu (0, 0.00%, 0, 0.00%, 0.18 ms, 1.71%)
  maxpool (0, 0.00%, 0, 0.00%, 0.23 ms, 2.19%)
  layer1 (0, 0.00%, 411.04 M, 10.03%, 1.45 ms, 13.78%)
    layer1.0 (0, 0.00%, 205.52 M, 5.01%, 0.73 ms, 6.94%)
      layer1.0.conv1 (0, 0.00%, 51.38 M, 1.25%, 0.15 ms, 1.43%)
      layer1.0.bn1 (0, 0.00%, 0, 0.00%, 0.11 ms, 1.05%)
      layer1.0.relu (0, 0.00%, 0, 0.00%, 0.09 ms, 0.86%)
      layer1.0.conv2 (0, 0.00%, 51.38 M, 1.25%, 0.16 ms, 1.52%)
      ...
  layer2 (0, 0.00%, 822.08 M, 20.05%, 2.34 ms, 22.24%)
  layer3 (0, 0.00%, 1.64 G, 40.01%, 3.67 ms, 34.89%)
  layer4 (0, 0.00%, 822.08 M, 20.05%, 1.54 ms, 14.64%)
  avgpool (0, 0.00%, 0, 0.00%, 0.08 ms, 0.76%)
  fc (2.05 M, 8.01%, 0, 0.00%, 0.12 ms, 1.14%)

------------------------------------------------------------------------------
```

### 6.6 Analyzing FLOPS Results

Let's break down the key metrics:

#### 1. FLOPs of the Model

```
fwd FLOPs per GPU: 8.20 G (GigaFLOPs)
```

**What this means:**
- One forward pass requires 8.2 billion floating-point operations
- This is fixed for ResNet50 at this batch size
- Doubling batch size doubles FLOPs

#### 2. Forward Pass FLOPS (Throughput)

```
fwd FLOPS per GPU: 779.47 GFLOPS
```

**What this means:**
- GPU is executing 779 billion FLOPs per second during forward pass
- This is achieved performance, not theoretical

#### 3. Efficiency Calculation

```
Theoretical peak (MI325X FP32): ~163,000 GFLOPS (163 TFLOPS)
Achieved: 779.47 GFLOPS
Efficiency: (779.47 / 163,000) × 100% = 0.48%
```

**Wait, only 0.48%?! Is this bad?**

Not necessarily! Here's why:

- **Small batch size:** BS=32 doesn't saturate the GPU
- **Mixed operations:** Not all operations are compute-intensive
- **Memory bound:** Some operations are limited by memory bandwidth, not compute

Let's verify this with a larger batch:

### 6.7 Batch Size Impact on Efficiency

Run with different batch sizes:

```bash
# Small batch
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 16 --iterations 20 --flops-prof-step 10 | grep "fwd FLOPS"

# Medium batch
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --flops-prof-step 10 | grep "fwd FLOPS"

# Large batch
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 20 --flops-prof-step 10 | grep "fwd FLOPS"
```

**Create a table:**

| Batch Size | FLOPs per Forward (G) | Achieved GFLOPS | Efficiency (%) |
|------------|----------------------|-----------------|----------------|
| 16         | ?                    | ?               | ?              |
| 32         | 8.20                 | 779.47          | 0.48%          |
| 64         | ?                    | ?               | ?              |
| 128        | ?                    | ?               | ?              |

**What pattern do you see?**
- Larger batches → Higher achieved GFLOPS
- FLOPs per forward increases linearly with batch size
- Efficiency improves with batch size

### 6.8 FP16 FLOPS Comparison

Let's see how FP16 affects FLOPs efficiency:

```bash
# FP32
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --fp16 0 --flops-prof-step 10 | grep "fwd FLOPS"

# FP16
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --fp16 1 --flops-prof-step 10 | grep "fwd FLOPS"
```

**Compare:**
```
FP32 achieved GFLOPS: ___________ GFLOPS
FP16 achieved GFLOPS: ___________ GFLOPS

FP32 peak (MI325X): 163 TFLOPS
FP16 peak (MI325X): 653 TFLOPS

FP32 efficiency: (___________ / 163,000) × 100% = ___________%
FP16 efficiency: (___________ / 653,000) × 100% = ___________%
```

### 6.9 Identifying Bottleneck Layers

From the detailed profile, look at "fwd latency":

```
Top modules by forward latency:
  10.52 ms    |  ResNet (total)
   3.67 ms    |  layer3 (34.89% of total!)
   2.34 ms    |  layer2 (22.24% of total)
   1.54 ms    |  layer4
   1.45 ms    |  layer1
```

**Analysis:**
- **layer3 is the bottleneck** (35% of forward time!)
- This makes sense: layer3 has the most FLOPs (1.64 G, 40% of total)
- Optimizing layer3 would have the biggest impact

### 6.10 Compute vs Memory Bound Analysis

Let's determine if ResNet50 is compute-bound or memory-bound:

#### Arithmetic Intensity Calculation

```
Arithmetic Intensity = FLOPs / Bytes Transferred

For ResNet50 forward pass:
- FLOPs: 8.20 G
- Weights: 25.56 M params × 4 bytes = 102 MB
- Activations: ~4 GB (estimated)
- Total bytes: ~4.1 GB

Arithmetic Intensity = 8.20 G / 4.1 GB ≈ 2.0 FLOPs/byte
```

**Interpretation:**

```
Arithmetic Intensity (FLOPs/byte):
< 1:   Severely memory-bound
1-10:  Memory-bound (typical for ResNet)
10-50: Balanced
> 50:  Compute-bound
```

**ResNet50 is memory-bound!** This explains the low efficiency.

**Optimization strategies:**
- Increase batch size (amortize memory transfers)
- Use FP16 (reduce bytes transferred)
- Fuse operations (reduce intermediate tensors)
- Use better memory layouts

### 6.11 Checkpoint

Before continuing:

- [ ] You understand what FLOPs and GFLOPS mean
- [ ] You can measure achieved GFLOPS with DeepSpeed profiler
- [ ] You understand efficiency = achieved / theoretical
- [ ] You know the difference between compute-bound and memory-bound
- [ ] You can identify bottleneck layers
- [ ] You understand why ResNet50 has low efficiency

**Record your FLOPS results in `my_workshop_results.txt`!**

---

## 7. Exercise 5: Multi-GPU Scaling

### 7.1 Objective

Scale your inference workload across multiple GPUs using distributed data parallel.

**What you'll learn:**
- How to use `torchrun` for multi-GPU execution
- Understanding data parallelism
- Measuring scaling efficiency
- Common multi-GPU issues

### 7.2 What is Distributed Data Parallel (DDP)?

**Data Parallelism:**
- Split batch across multiple GPUs
- Each GPU has a complete copy of the model
- Process different data on each GPU in parallel
- Combine results at the end

**Example with 2 GPUs:**
```
Original batch: 64 samples
├── GPU 0: processes samples 0-31
└── GPU 1: processes samples 32-63

Throughput: ~2x faster (ideally)
```

**Key concepts:**
- **World Size:** Total number of processes (= number of GPUs)
- **Rank:** ID of current process (0 to world_size-1)
- **Local Rank:** ID of GPU on current node

### 7.3 Prerequisites: Check Available GPUs

```bash
# Check how many GPUs you have
rocm-smi --showid

# Should show something like:
# GPU[0] : GPU ID: 0
# GPU[1] : GPU ID: 1
# ...
```

**For this exercise, you need at least 2 GPUs.**

If you only have 1 GPU, you can still read along and understand the concepts!

### 7.4 Single GPU Baseline (For Comparison)

First, establish a single-GPU baseline:

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 20
```

**Record the throughput:**
```
Single GPU (BS=64): ___________ samples/sec
```

### 7.5 Running with 2 GPUs

Now let's scale to 2 GPUs:

```bash
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 128 \
    --iterations 20
```

**Important notes:**
- `--nproc-per-node 2`: Use 2 GPUs
- `--batch-size 128`: Total batch size (64 per GPU)
- `torchrun` automatically splits the batch

**Expected output:**
```
**** Launching with torchrun ****
Setting up process group...
[GPU 0] Initializing...
[GPU 1] Initializing...
Process group initialized.

[GPU 0] Using network: resnet50
[GPU 0] Local batch size: 64
[GPU 0] Global batch size: 128
[GPU 1] Using network: resnet50
[GPU 1] Local batch size: 64
[GPU 1] Global batch size: 128

Warming up...
[GPU 0] Warmup complete.
[GPU 1] Warmup complete.

Running timed iterations...
[GPU 0] Epoch 0: Loss = 6.9078, Time = 0.063 seconds
[GPU 1] Epoch 0: Loss = 6.9078, Time = 0.063 seconds
...

========================================
Performance Summary (GPU 0):
========================================
Global batch size: 128
Local batch size: 64
World size: 2

Average time per iteration: 0.063 seconds
Throughput: 2032.5 samples/sec (global)
Per-GPU throughput: 1016.3 samples/sec
GPU Memory Usage: 4523 MB
========================================
```

### 7.6 Analyzing Multi-GPU Results

Let's calculate scaling efficiency:

```
Single GPU:  ___________ samples/sec (BS=64)
Two GPUs:    ___________ samples/sec (BS=128)

Ideal 2-GPU: ___________ × 2 = ___________ samples/sec
Actual 2-GPU: ___________ samples/sec

Scaling efficiency: (Actual / Ideal) × 100% = ___________%
```

**Typical results:**
- **Perfect scaling (100%):** Rare! Means no overhead
- **Good scaling (90-95%):** Common for large batches
- **Moderate scaling (80-90%):** Typical for medium batches
- **Poor scaling (<80%):** Communication overhead, small batches

### 7.7 Scaling Factors: What Affects Efficiency?

#### 1. Batch Size Per GPU

```bash
# Small batch per GPU (32)
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20

# Medium batch per GPU (64)
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 20

# Large batch per GPU (128)
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 20
```

**Create a table:**

| Batch per GPU | Total Batch | 1-GPU Throughput | 2-GPU Throughput | Scaling Efficiency |
|---------------|-------------|------------------|------------------|--------------------|
| 32            | 64          | ?                | ?                | ?%                 |
| 64            | 128         | ?                | ?                | ?%                 |
| 128           | 256         | ?                | ?                | ?%                 |

**Pattern:**
- Larger batches → Better scaling efficiency
- Why? Communication overhead is amortized

#### 2. Model Size

```bash
# Small model (ResNet18)
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet18 --batch-size 128 --iterations 20

# Medium model (ResNet50)
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 20

# Large model (ResNet152)
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet152 --batch-size 128 --iterations 20
```

**Observation:**
- Larger models scale better
- Why? More computation relative to communication

### 7.8 Running with 4 GPUs (If Available)

If you have 4+ GPUs:

```bash
# 4 GPUs
torchrun --nproc-per-node 4 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 256 \
    --iterations 20
```

**Scaling analysis:**

| GPUs | Batch Size | Throughput | Ideal | Efficiency |
|------|------------|------------|-------|------------|
| 1    | 64         | ___        | ___   | 100%       |
| 2    | 128        | ___        | ___   | ___%       |
| 4    | 256        | ___        | ___   | ___%       |

**Typical pattern:**
- 1 → 2 GPUs: 90-95% efficiency
- 2 → 4 GPUs: 85-90% efficiency
- Efficiency decreases with more GPUs (communication overhead)

### 7.9 Common Multi-GPU Issues

#### Issue 1: "RuntimeError: NCCL error"

```bash
# Solution 1: Check GPU visibility
export ROCR_VISIBLE_DEVICES=0,1

# Solution 2: Set NCCL debug level
export NCCL_DEBUG=INFO
```

#### Issue 2: "OOM on some GPUs but not others"

**Cause:** Imbalanced workload or initialization

```bash
# Check memory on all GPUs
rocm-smi

# Should be similar across GPUs
```

#### Issue 3: "Very poor scaling (<50%)"

**Possible causes:**
- Batch size too small per GPU
- High communication overhead
- CPU bottleneck
- Slow interconnect

**Debug steps:**
```bash
# 1. Profile a single GPU
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20

# 2. Check if single GPU is efficient
# If single GPU is slow, fix that first!

# 3. Increase batch size per GPU
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 256 --iterations 20
```

#### Issue 4: "Hangs at initialization"

```bash
# Check if processes can communicate
export NCCL_DEBUG=INFO
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 2

# Look for NCCL initialization messages
# If stuck, check firewall, network, GPU interconnect
```

### 7.10 Best Practices for Multi-GPU Inference

**1. Batch Size:**
- Use largest batch that fits in memory per GPU
- Larger batches = better scaling

**2. Model Loading:**
- Load model once, copy to all GPUs
- Don't load from disk on each GPU (slow!)

**3. Data Loading:**
- Use multiple workers for data loading
- Pre-fetch batches to avoid GPU idle time

**4. Warmup:**
- Always warmup before timing
- First iteration compiles kernels

**5. Synchronization:**
- Use `torch.cuda.synchronize()` when timing
- Otherwise you measure launch time, not execution time

### 7.11 Checkpoint

Before continuing:

- [ ] You can use `torchrun` for multi-GPU execution
- [ ] You understand batch splitting in DDP
- [ ] You can calculate scaling efficiency
- [ ] You understand factors affecting scaling
- [ ] You know how to debug common multi-GPU issues

**Record your multi-GPU results in `my_workshop_results.txt`!**

---

## 8. Exercise 6: PyTorch 2.0 Compilation

### 8.1 Objective

Use PyTorch 2.0's `torch.compile` to automatically optimize your model.

**What you'll learn:**
- What is torch.compile and how it works
- Different compilation modes
- Measuring speedup from compilation
- When compilation helps (and when it doesn't)

### 8.2 What is torch.compile?

**PyTorch 2.0 introduced `torch.compile`:**
- Analyzes your model's computation graph
- Applies graph-level optimizations
- Generates optimized GPU kernels
- No code changes required!

**How it works:**
```
1. Trace your model: Record operations
2. Optimize graph: Fuse operations, eliminate redundancy
3. Generate kernels: Compile optimized CUDA/ROCm code
4. Execute: Run optimized version
```

**Potential speedups:**
- Operator fusion (reduce kernel launches)
- Memory layout optimization
- Kernel specialization
- Dead code elimination

### 8.3 Baseline (No Compilation)

First, run without compilation:

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 20
```

**Record baseline:**
```
No compilation: ___________ samples/sec
```

### 8.4 Default Compilation Mode

Now enable compilation with default settings:

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 20 \
    --compile
```

**Note:** First run will be SLOW (compilation time!)

**Expected output:**
```
Using network: resnet50
Batch size: 64
Iterations: 20
PyTorch Compile: ENABLED (mode=default)

Compiling model...
[Compiling...] This may take 1-2 minutes on first run...
[COMPILE] Tracing model...
[COMPILE] Optimizing graph...
[COMPILE] Generating kernels...
Compilation complete.

Warming up...
Warmup complete.

Running timed iterations...
Epoch 0: Loss = 6.9078, Time = 0.058 seconds
...

========================================
Performance Summary:
========================================
Throughput: 1103.4 samples/sec
Compilation time: 87.3 seconds (first run only)
========================================
```

### 8.5 Understanding Compilation Overhead

**First run:**
- Slow! Compilation takes 1-3 minutes
- Not included in performance measurements

**Subsequent runs:**
- Fast! Cached kernels are reused
- No recompilation needed

**When is this worth it?**
- Production deployments (compile once, run millions of times)
- Long-running inference servers
- Batch processing large datasets

**When is it NOT worth it?**
- Single inference runs
- Prototyping
- Frequently changing models

### 8.6 Compilation Modes

PyTorch 2.0 has different compilation modes:

#### Mode 1: default (Conservative)

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 20 \
    --compile
```

**Characteristics:**
- Fast compilation
- Safe optimizations
- Moderate speedup

#### Mode 2: reduce-overhead

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 20 \
    --compile \
    --compileContext "{'mode': 'reduce-overhead'}"
```

**Characteristics:**
- Focus on reducing Python overhead
- Faster for many small operations
- Good for models with lots of layers

#### Mode 3: max-autotune (Aggressive)

```bash
python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 64 \
    --iterations 20 \
    --compile \
    --compileContext "{'mode': 'max-autotune'}"
```

**Characteristics:**
- VERY slow compilation (5-10 minutes!)
- Tries many kernel variants
- Benchmarks each variant
- Selects fastest
- Best runtime performance

**Expected output:**
```
[COMPILE] Mode: max-autotune
[COMPILE] Testing kernel variant 1/53...
[COMPILE] Testing kernel variant 2/53...
[COMPILE] Testing kernel variant 3/53...
...
[COMPILE] Best kernel selected: variant 27
Compilation complete (took 347.2 seconds).

Throughput: 1287.5 samples/sec  ← Even faster!
```

### 8.7 Comparing Compilation Modes

Run all modes and compare:

```bash
# No compilation
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 > results_no_compile.txt

# Default mode
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --compile > results_default.txt

# Reduce overhead
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --compile --compileContext "{'mode': 'reduce-overhead'}" > results_reduce_overhead.txt

# Max autotune (WARNING: This takes 5-10 minutes!)
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --compile --compileContext "{'mode': 'max-autotune'}" > results_max_autotune.txt
```

**Extract throughput:**
```bash
grep "Throughput" results_no_compile.txt
grep "Throughput" results_default.txt
grep "Throughput" results_reduce_overhead.txt
grep "Throughput" results_max_autotune.txt
```

**Create comparison table:**

| Mode | Compilation Time | Throughput | Speedup |
|------|------------------|------------|---------|
| No compile | 0 seconds | ___ samples/sec | 1.0x |
| default | ___ seconds | ___ samples/sec | ___x |
| reduce-overhead | ___ seconds | ___ samples/sec | ___x |
| max-autotune | ___ seconds | ___ samples/sec | ___x |

**Typical results:**
- default: 1.1-1.2x speedup
- reduce-overhead: 1.1-1.3x speedup
- max-autotune: 1.2-1.4x speedup

### 8.8 When Does Compilation Help Most?

Let's test different models:

```bash
# ResNet18 (small model)
python3 micro_benchmarking_pytorch.py --network resnet18 --batch-size 64 --iterations 20
python3 micro_benchmarking_pytorch.py --network resnet18 --batch-size 64 --iterations 20 --compile

# ResNet50 (medium model)
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --compile

# ResNet152 (large model)
python3 micro_benchmarking_pytorch.py --network resnet152 --batch-size 64 --iterations 20
python3 micro_benchmarking_pytorch.py --network resnet152 --batch-size 64 --iterations 20 --compile
```

**Pattern:**
- Deeper models (more layers) → More benefit from compilation
- Why? More opportunities for fusion and optimization

### 8.9 Compilation + FP16

Let's combine compilation with FP16:

```bash
# FP32 no compile
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --fp16 0

# FP32 with compile
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --fp16 0 --compile

# FP16 no compile
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --fp16 1

# FP16 with compile
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --fp16 1 --compile
```

**Comparison table:**

| Configuration | Throughput | Speedup vs FP32 No Compile |
|---------------|------------|---------------------------|
| FP32, No compile | ___ | 1.0x |
| FP32, Compiled | ___ | ___x |
| FP16, No compile | ___ | ___x |
| FP16, Compiled | ___ | ___x |

**Best combination:** FP16 + max-autotune compilation!

### 8.10 Common Compilation Issues

#### Issue 1: "RuntimeError: Compiled function failed"

**Cause:** Compilation doesn't support some operations

**Solution:**
```bash
# Disable compilation for troubleshooting
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20
```

#### Issue 2: "Very slow compilation (>10 minutes)"

**Cause:** max-autotune mode tests many variants

**Solution:**
- Use `default` mode for faster compilation
- Only use `max-autotune` for production
- Be patient! It's worth it for long-running inference

#### Issue 3: "No speedup from compilation"

**Possible causes:**
- Model already well-optimized
- Bottleneck is memory, not compute
- Batch size too small

**Debug:**
```bash
# Try larger batch
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 20 --compile

# Try different model
python3 micro_benchmarking_pytorch.py --network efficientnet_b0 --batch-size 64 --iterations 20 --compile
```

### 8.11 Checkpoint

Before continuing:

- [ ] You understand what torch.compile does
- [ ] You can enable compilation with `--compile`
- [ ] You know the different compilation modes
- [ ] You understand compilation overhead (first run)
- [ ] You can combine compilation with FP16
- [ ] You know when compilation helps most

**Record your compilation results in `my_workshop_results.txt`!**

---

## 9. Exercise 7: ROCm Profiler Integration

### 9.1 Objective

Use ROCm-specific profilers for deep kernel-level analysis.

**What you'll learn:**
- Using `rocprof` for kernel statistics
- Using `rocprofv2` for timeline visualization
- Interpreting kernel-level metrics
- Identifying GPU inefficiencies

### 9.2 ROCm Profiling Tools Overview

| Tool | Purpose | Output |
|------|---------|--------|
| **rocprof** | Kernel statistics (CSV) | Execution times, call counts |
| **rocprofv2** | Timeline visualization | JSON for Perfetto UI |
| **rocprof-compute** | Hardware counters | Memory bandwidth, occupancy |

**When to use each:**
1. Start with manual timing (Exercise 1)
2. Use PyTorch Profiler for operator-level (Exercise 3)
3. Use `rocprof` for kernel statistics (this exercise)
4. Use `rocprofv2` for timeline analysis (this exercise)
5. Use `rocprof-compute` for advanced optimization (advanced users)

### 9.3 Using rocprof for Kernel Statistics

#### Step 1: Run with rocprof

```bash
rocprof --stats python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 10
```

**Note:** Reduced iterations to keep profile size manageable



**Expected output:**
```
ROCProfiler: Profiling enabled
Profiling output will be in: results.csv

Running benchmark...
[... normal benchmark output ...]

Profiling complete.
Results saved to: results.csv
```

#### Step 2: Examine the results

```bash
# View first 20 lines
head -20 results.csv

# Or open in spreadsheet program
# LibreOffice, Excel, etc.
```

**Sample results.csv:**
```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP0_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PC0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT8_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS64_WG32_16_1_WGM8",42,"2476543000","58965310","45.67%"
"void at::native::(anonymous namespace)::batch_norm_collect_statistics_kernel<float, float, int, 128>(at::native::(anonymous namespace)::BatchNormCollectStatisticsKernelParams<float, float, int>)",80,"523456000","6543200","9.65%"
"void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::MulFunctor<float> >, at::detail::Array<char*, 3> >(int, at::native::BinaryFunctor<float, float, float, at::native::MulFunctor<float> >, at::detail::Array<char*, 3>)",320,"387234000","1210106","7.14%"
...
```

#### Step 3: Analyze kernel statistics

```bash
# Sort by total duration (slowest kernels)
sort -t',' -k3 -nr results.csv | head -20

# Count total kernel launches
wc -l results.csv

# Find memory copy operations
grep -i "memcpy" results.csv
```

### 9.4 Understanding Kernel Statistics

Let's break down the CSV columns:

#### 1. Name
- Kernel function name
- Long, mangled names (C++ name mangling)
- Look for keywords: `conv`, `gemm`, `batch_norm`, `relu`

#### 2. Calls
- Number of times kernel was launched
- High call count might indicate opportunity for fusion

#### 3. TotalDurationNs
- Total time spent in this kernel (nanoseconds)
- Sort by this to find bottlenecks!

#### 4. AverageNs
- Average time per kernel launch
- `TotalDurationNs / Calls`

#### 5. Percentage
- Percentage of total GPU time
- Sum of top 5-10 kernels often 80-90% of total time

### 9.5 Hands-On Analysis

Using your `results.csv`, answer:

**Question 1:** What is the slowest kernel?
```
Name: _______________________________________
Total Duration: _____________ ms (divide ns by 1,000,000)
Percentage: _____________%
```

**Question 2:** How many total kernel launches?
```
Total kernels: _____________ (use: wc -l results.csv)
```

**Question 3:** What percentage of time is spent in top 5 kernels?
```
Kernel 1: ______________%
Kernel 2: ______________%
Kernel 3: ______________%
Kernel 4: ______________%
Kernel 5: ______________%

Total: ______________%
```

**Question 4:** Are there memory copy operations?
```
grep -i "memcpy" results.csv

Found: _______ memcpy operations
Total time: _______ ms
Percentage: _______%
```

**Interpretation:**
- If memcpy > 10%: Memory transfer is a bottleneck
- If memcpy < 5%: Compute-bound, memory transfers are efficient

### 9.6 Comparing FP32 vs FP16 Kernels

Let's see how kernels differ:

```bash
# FP32
rocprof --stats -o profile_fp32.csv python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 10 --fp16 0

# FP16
rocprof --stats -o profile_fp16.csv python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 10 --fp16 1
```

#### Compare kernel counts:

```bash
# FP32 kernel count
wc -l profile_fp32.csv

# FP16 kernel count
wc -l profile_fp16.csv
```

#### Compare slowest kernel:

```bash
# FP32 slowest
sort -t',' -k3 -nr profile_fp32.csv | head -2

# FP16 slowest
sort -t',' -k3 -nr profile_fp16.csv | head -2
```

**Create comparison:**
```
FP32:
  Total kernels: _____________
  Slowest kernel: _____________ ms

FP16:
  Total kernels: _____________
  Slowest kernel: _____________ ms

Speedup: _____________ / _____________ = _____________x
```

### 9.7 Using rocprofv2 for Timeline Visualization

Now let's create a timeline visualization:

```bash
rocprofv2 --kernel-trace -o timeline.json python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 5
```

**Note:** Only 5 iterations to keep file size small

**Expected output:**
```
ROCProfiler v2: Timeline tracing enabled
Output file: timeline.json

Running benchmark...
[... normal benchmark output ...]

Timeline saved to: timeline.json
File size: 23.4 MB

View at: https://ui.perfetto.dev
```

#### Step 2: Visualize the timeline

1. **Open Chrome browser**
2. **Go to:** `https://ui.perfetto.dev`
3. **Click "Open trace file"**
4. **Select `timeline.json`**

**You'll see a timeline view!**

### 9.8 Interpreting the Timeline

The timeline shows:

**X-axis:** Time (microseconds)
**Y-axis:** Different "tracks":
- CPU threads
- GPU streams
- Kernel executions
- Memory copies

**What to look for:**

#### 1. GPU Idle Time (Gaps)
```
Good:  ████████████████████████████████  (No gaps, fully utilized)
Bad:   ███  ██  ███  ██  ███  ██  ███  (Lots of gaps, idle time)
```

**If you see gaps:**
- CPU bottleneck (slow data loading, Python overhead)
- Synchronization issues
- Small batch size

#### 2. Kernel Duration Variance
```
Good:  ████ ████ ████ ████ ████  (Consistent duration)
Bad:   █ ████ ██ ████████ █ ████  (Highly variable)
```

**If highly variable:**
- Different batch sizes
- Conditional execution
- Autotuning happening

#### 3. Memory Copies
```
Look for: Memcpy HtoD (Host to Device)
          Memcpy DtoH (Device to Host)
```

**If significant:**
- Consider pinned memory
- Use async copies
- Overlap compute and transfer

#### 4. Kernel Launch Overhead
```
Measure gap between kernel end and next kernel start
```

**If large gaps (>10μs):**
- Kernel fusion opportunity
- CPU-side overhead

### 9.9 Advanced: rocprof-compute Metrics

For advanced users, `rocprof-compute` provides hardware counters:

```bash
rocprof-compute profile -w profile.csv python3 micro_benchmarking_pytorch.py \
    --network resnet50 \
    --batch-size 32 \
    --iterations 5
```

**Metrics available:**
- Memory bandwidth utilization (%)
- GPU occupancy (%)
- Cache hit rates
- Arithmetic intensity
- Wave occupancy

**Example metrics:**
```
LDS Bank Conflicts: 234
L2 Cache Hit Rate: 87.5%
Memory Bandwidth Util: 72.3%
Wave Occupancy: 45.2%
```

**Interpretation:**
- Memory bandwidth > 80%: Memory-bound
- Occupancy < 30%: Poor kernel utilization
- Cache hit < 70%: Poor memory access patterns

### 9.10 Checkpoint

Before continuing:

- [ ] You can use `rocprof --stats` for kernel statistics
- [ ] You can identify slowest kernels
- [ ] You can count kernel launches
- [ ] You can use `rocprofv2` for timeline visualization
- [ ] You can interpret timeline traces
- [ ] You understand GPU idle time, gaps, and kernel duration

**Record your profiling insights in `my_workshop_results.txt`!**

---

## 10. Wrap-up & Best Practices

### 10.1 Workshop Summary

Congratulations! You've completed the ROCm PyTorch Inference Benchmark Workshop!

**What you've learned:**

1. **Environment Setup**
   - Verify ROCm, PyTorch, GPUs
   - Run standardized benchmarks

2. **Benchmark Tool Mastery**
   - Use `micro_benchmarking_pytorch.py`
   - Understand command-line options
   - Interpret output metrics

3. **Precision Optimization**
   - FP16 vs FP32 comparison
   - 2x speedup, 40% memory reduction
   - When to use FP16

4. **Framework Profiling**
   - PyTorch Profiler for operator-level analysis
   - DeepSpeed FLOPS profiler for efficiency
   - Identifying bottleneck operations

5. **Multi-GPU Scaling**
   - Distributed data parallel with `torchrun`
   - Scaling efficiency calculation
   - Debugging multi-GPU issues

6. **Compilation Optimization**
   - torch.compile for automatic optimization
   - Different compilation modes
   - 1.2-1.4x additional speedup

7. **Hardware Profiling**
   - rocprof for kernel statistics
   - rocprofv2 for timeline visualization
   - Finding GPU inefficiencies

### 10.2 Performance Optimization Checklist

Use this checklist for optimizing YOUR models:

#### Phase 1: Baseline & Measurement
- [ ] Establish baseline performance (no optimizations)
- [ ] Use manual timing with `torch.cuda.synchronize()`
- [ ] Record throughput, latency, memory usage
- [ ] Run multiple iterations for stable measurements

#### Phase 2: Low-Hanging Fruit
- [ ] Use FP16 if model supports it (2x speedup typical)
- [ ] Increase batch size to maximum (better GPU utilization)
- [ ] Enable `torch.compile` with default mode (1.2x speedup typical)
- [ ] Use `model.eval()` and `torch.no_grad()` for inference

#### Phase 3: Profiling
- [ ] PyTorch Profiler: Identify slow operators
- [ ] rocprof: Find bottleneck kernels
- [ ] rocprofv2: Visualize timeline, find idle time
- [ ] DeepSpeed FLOPS: Calculate efficiency

#### Phase 4: Optimization
- [ ] If memory-bound (<20% efficiency):
  - Increase batch size
  - Use FP16
  - Fuse operations
  - Optimize memory layout

- [ ] If compute-bound (>40% efficiency):
  - Use specialized kernels (cuDNN/MIOpen)
  - Try custom Triton kernels
  - Use torch.compile max-autotune

- [ ] If CPU-bound (gaps in timeline):
  - Use data loading workers
  - Pre-allocate tensors
  - Reduce Python overhead
  - Use JIT compilation

#### Phase 5: Validation
- [ ] Re-measure performance
- [ ] Verify numerical accuracy (compare outputs)
- [ ] Test with different batch sizes
- [ ] Ensure consistent results (low std dev)

#### Phase 6: Scaling (If Multi-GPU)
- [ ] Test single GPU first
- [ ] Scale to 2, 4, 8 GPUs
- [ ] Calculate scaling efficiency
- [ ] Optimize batch size per GPU

### 10.3 Common Pitfalls and How to Avoid Them

#### Pitfall 1: Not Using torch.cuda.synchronize()

**Problem:**
```python
start = time.time()
output = model(input)
end = time.time()  # WRONG! GPU is still running
```

**Solution:**
```python
start = time.time()
output = model(input)
torch.cuda.synchronize()  # Wait for GPU to finish
end = time.time()
```

#### Pitfall 2: Including Warmup in Measurements

**Problem:**
```python
for i in range(20):
    output = model(input)
# Average includes slow first iteration
```

**Solution:**
```python
# Warmup
for i in range(5):
    output = model(input)
torch.cuda.synchronize()

# Timed iterations
start = time.time()
for i in range(20):
    output = model(input)
torch.cuda.synchronize()
end = time.time()  # Excludes warmup
```

#### Pitfall 3: Batch Size Too Small

**Problem:**
- Low GPU utilization
- High kernel launch overhead
- Poor performance

**Solution:**
- Increase batch size
- Profile to find optimal batch size
- Trade-off: Larger batch = more memory, higher throughput

#### Pitfall 4: Ignoring Numerical Accuracy

**Problem:**
- FP16 causes NaN or Inf
- Model outputs are wrong
- Silent numerical errors

**Solution:**
```python
# Always verify outputs
output_fp32 = model_fp32(input)
output_fp16 = model_fp16(input)

diff = (output_fp32 - output_fp16).abs().max()
print(f"Max difference: {diff}")  # Should be < 0.01
```

#### Pitfall 5: Over-Optimizing Small Operations

**Problem:**
- Spend hours optimizing 2% of runtime
- Ignore operations that take 80% of time

**Solution:**
- Profile first!
- Focus on bottlenecks (top 80% of time)
- Use Pareto principle: 20% of operations take 80% of time

### 10.4 When to Use Each Technique

| Technique | Speedup | Effort | When to Use |
|-----------|---------|--------|-------------|
| FP16 | 2x | Low (1 line) | Almost always for inference |
| Larger batch | 1.5-3x | Low | When memory allows |
| torch.compile | 1.2-1.4x | Low (1 line) | Production deployments |
| Multi-GPU | Nx | Medium | Large throughput requirements |
| Custom kernels | 2-10x | High | Bottleneck operations |
| Model optimization | 2-5x | High | Production, critical latency |

### 10.5 Real-World Deployment Recommendations

#### For Production Inference:

1. **Model Optimization:**
   - Use FP16 or INT8 quantization
   - Compile with max-autotune mode
   - Prune unnecessary operations

2. **Batch Processing:**
   - Use largest batch size that meets latency requirements
   - Implement dynamic batching (combine requests)

3. **Hardware Selection:**
   - Profile your specific model on different GPUs
   - Consider memory requirements
   - Calculate cost per inference

4. **Monitoring:**
   - Track throughput, latency, memory usage
   - Set up alerts for performance degradation
   - Log profiling data periodically

5. **Optimization Cycle:**
   - Measure → Analyze → Optimize → Validate
   - Repeat as workload changes
   - Keep profiling infrastructure in place

### 10.6 Resources for Further Learning

#### Official Documentation
- **ROCm Documentation:** https://rocm.docs.amd.com/
- **PyTorch Profiler:** https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **DeepSpeed:** https://www.deepspeed.ai/tutorials/flops-profiler/
- **torch.compile:** https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

#### Profiling Tools
- **rocprof Guide:** https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
- **rocprofv2:** https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
- **Perfetto UI:** https://ui.perfetto.dev

#### Community
- **PyTorch Forums:** https://discuss.pytorch.org/
- **ROCm GitHub:** https://github.com/RadeonOpenCompute/ROCm
- **AMD Developer Community:** https://community.amd.com/

### 10.7 Next Steps

**Immediate Actions:**
1. Apply techniques to YOUR models
2. Establish baselines for your workload
3. Create profiling scripts for regular testing
4. Document optimization wins

**Short-term (1-2 weeks):**
1. Deep-dive into your bottleneck operations
2. Try custom optimizations (if needed)
3. Test multi-GPU scaling (if applicable)
4. Implement monitoring

**Long-term (1-3 months):**
1. Build optimization into CI/CD
2. Create performance regression tests
3. Track performance over time
4. Share learnings with team

### 10.8 Workshop Feedback

**Please provide feedback on:**

1. **What worked well?**
   - Which exercises were most valuable?
   - What concepts were clearest?

2. **What could be improved?**
   - Which parts were confusing?
   - What needs more detail?

3. **What's missing?**
   - Topics you wanted to cover?
   - Tools or techniques?

4. **Overall experience:**
   - Pacing (too fast/slow)?
   - Difficulty level?
   - Practical applicability?

### 10.9 Final Checklist

Before leaving the workshop:

- [ ] All exercises completed
- [ ] Results recorded in `my_workshop_results.txt`
- [ ] Understood key concepts (FP16, profiling, multi-GPU, compilation)
- [ ] Know how to profile YOUR models
- [ ] Have resources for further learning
- [ ] Can apply techniques to production workloads

### 10.10 Thank You!

**Congratulations on completing the ROCm PyTorch Inference Benchmark Workshop!**

You now have the skills to:
- Benchmark AI models systematically
- Use profiling tools to find bottlenecks
- Apply optimization techniques
- Scale workloads across GPUs
- Measure and validate improvements

**Go forth and optimize!** 

---

## Appendix A: Quick Reference Commands

### Basic Benchmarking
```bash
# Single GPU, FP32
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20

# Single GPU, FP16
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --fp16 1

# Multi-GPU
torchrun --nproc-per-node 2 micro_benchmarking_pytorch.py --network resnet50 --batch-size 128 --iterations 20

# With compilation
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 64 --iterations 20 --compile
```

### Profiling
```bash
# PyTorch Profiler
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 10 --autograd-profiler

# DeepSpeed FLOPS
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 20 --flops-prof-step 10

# rocprof statistics
rocprof --stats python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 10

# rocprofv2 timeline
rocprofv2 --kernel-trace -o timeline.json python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 5
```

### System Checks
```bash
# Check ROCm
rocminfo | grep "Name:"

# Check GPU
rocm-smi

# Check PyTorch
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Check GPU memory
rocm-smi --showmeminfo vram
```

---

## Appendix B: Troubleshooting Guide

### GPU Not Detected
```bash
# Check GPU visibility
rocminfo | grep "Name:"

# Check permissions
sudo usermod -aG video $USER
sudo usermod -aG render $USER
# Logout and login

# Verify
groups | grep video
```

### Out of Memory (OOM)
```bash
# Reduce batch size
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 16 --iterations 20

# Use FP16
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 20 --fp16 1

# Clear cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

### Poor Performance
```bash
# Check GPU utilization during run
watch -n 0.5 rocm-smi

# Should show ~100% utilization
# If low, check:
# 1. Batch size too small
# 2. CPU bottleneck
# 3. Thermal throttling
```

### Inconsistent Results
```bash
# Increase iterations for better averaging
python3 micro_benchmarking_pytorch.py --network resnet50 --batch-size 32 --iterations 50

# Check for system interference
top
# Look for other processes using CPU/GPU
```

---

**End of Workshop Guide**


**Exercises Completed:** 7 major exercises
**Skills Acquired:** GPU benchmarking, profiling, optimization

**Now go optimize your models!** 
