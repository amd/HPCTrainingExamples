
# Training Notes

README.md from `HPCTrainingExamples/MLExamples/inference_benchmark` in the Training Examples repository

## Basic Training Run

DenseNet121 with torch.compile and mixed precision (FP16):

```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1
```

## Profiling

### PyTorch Profiler (Kineto)

Generate Chrome trace with detailed kernel timeline:

```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1 --kineto --iterations 10
```

Output: `trace<step>.json` files (viewable in chrome://tracing)

Options:
- `--kineto`: Enable Kineto profiler (torch.profiler with Chrome trace export)
- `--iterations`: Number of iterations (profiler captures wait=1, warmup=2, active=2)

### PyTorch Autograd Profiler (ROCTX)

For use with ROCm profilers (rocprof):

```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1 --autograd_profiler
```

Enables ROCTX markers for correlation with GPU kernel timeline in rocprof traces.

### DeepSpeed FLOPS Profiler

Detailed FLOPS and memory analysis:

```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --fp16 1 --flops-prof-step 10 --iterations 20
```

Options:
- `--flops-prof-step`: Iteration at which to capture profile (0-based index)
- `--iterations`: Total iterations (must be > flops-prof-step)

Output includes:
- FLOPS per layer and operation type
- Memory bandwidth utilization
- Parameter count and activation memory
- Theoretical vs achieved performance

## Performance Tuning

### MIOpen Kernel Tuning

For optimal performance on AMD GPUs, enable MIOpen find mode:

```bash
export MIOPEN_FIND_ENFORCE=3
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1
```

First run generates performance database at `~/.config/miopen/`. Subsequent runs use cached kernels.

### Torch Compile Modes

Default compilation:
```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1
```

Maximum optimization:
```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1 \
    --compileContext "{'mode': 'max-autotune', 'fullgraph': 'True'}"
```

Memory and matmul optimization:
```bash
python micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1 \
    --compileContext "{'options': {'static-memory': 'True', 'matmul-padding': 'True'}}"
```

## Multi-GPU Training

### 4-GPU Run

```bash
torchrun --nproc-per-node 4 micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1
```

### 8-GPU Run

```bash
torchrun --nproc-per-node 8 micro_benchmarking_pytorch.py --network densenet121 --batch-size 2048 --compile --fp16 1
```

**Batch size behavior:**
- `--batch-size` specifies global batch size across all GPUs
- Each GPU processes `batch-size / nproc-per-node` samples
- Example: `--batch-size 2048` with 4 GPUs → 512 samples/GPU

### Multi-GPU Profiling

#### PyTorch Profiler (Kineto)

Profile 4-GPU run with trace export:
```bash
torchrun --nproc-per-node 4 micro_benchmarking_pytorch.py \
    --network densenet121 --batch-size 2048 --compile --fp16 1 \
    --kineto --iterations 10
```

Output: `trace<step>.json` per rank (4 files total)

#### DeepSpeed FLOPS Profiler

Multi-GPU FLOPS analysis:
```bash
torchrun --nproc-per-node 4 micro_benchmarking_pytorch.py \
    --network densenet121 --batch-size 2048 --fp16 1 \
    --flops-prof-step 10 --iterations 20
```

Profile captures per-GPU metrics at specified iteration.

## Metrics to Track

- Throughput (images/sec)
- GPU memory utilization (GB)
- Training time per iteration (ms)
- FLOPS efficiency (% of peak)
- Memory bandwidth saturation (% of theoretical)
- Kernel occupancy
- Compilation overhead (first iteration vs steady state)


