# DeepSpeed FlopsProfiler — PyTorch examples

> Shared guide for [`imagenet`](../../imagenet/PROFILING.md),
> [`minGPT-ddp`](../../minGPT-ddp/PROFILING.md), and [`FSDP2`](../../FSDP2/PROFILING.md).
> Read each example's *Ground rules* first.

`--flops` runs the DeepSpeed `get_model_profile` on the model **without** needing a
DeepSpeed engine (it works on any `nn.Module`), reporting FLOPs, MACs, params, and
per-module forward latency. This is the theoretical **compute ceiling** to compare
against the measured kernel time from [torch.profiler](torch-profiler.md) and the
[rocprof-compute](rocprof-compute.md) roofline.

## Run (1 GPU is enough — it profiles the model, not the run)

| Example | Command |
|---------|---------|
| imagenet | `torchrun --standalone --nproc_per_node=1 ddp_resnet_bench.py -a resnet50 -b 64 --flops` |
| minGPT-ddp | `torchrun --standalone --nproc_per_node=1 ddp_gpt_bench.py --n-layer 12 --n-embd 768 --n-head 12 --batch-size 8 --flops` |
| FSDP2 | `torchrun --standalone --nproc_per_node=1 fsdp2_bench.py --n-layers 16 --dim 1024 --n-heads 16 --seq-len 512 --batch-size 8 --flops` |

Verified outputs (MI300A):

```
# imagenet (resnet50, batch 64)
fwd MACs per GPU:  261.71 GMACs
FLOPS_PROFILE arch=resnet50 batch=64 flops=525.51 G macs=261.71 GMACs params=25.56 M

# minGPT-ddp (small config)
FLOPS_PROFILE model=gpt n_layer=4 n_embd=256 batch=4 block=512 flops=71 G macs=35.48 GMACs params=29.29 M

# FSDP2 (small config — dense model rebuilt on rank 0)
FLOPS_PROFILE model=transformer n_layers=4 dim=256 batch=4 seq=128 flops=11.89 G macs=5.94 GMACs params=19.57 M
```

**Interpretation.** For imagenet, 525.5 GFLOPs forward for a 64-image batch and
25.56 M params → `25.56M × 4 B = 102 MB` all-reduced per step in fp32 (the
`grad_allreduce=` field of the `RESULT` line). Combine with
[torch.profiler](torch-profiler.md) to get achieved TFLOP/s (`FLOPs / kernel_time`).

> **FSDP2 note.** Parameters are sharded `DTensor`s the profiler cannot introspect,
> so `--flops` rebuilds **one dense model on rank 0** — the right number, since FSDP
> changes how work is *distributed*, not the total FLOPs. For a config too large to
> fit dense on one GPU, reduce `--n-layers` for the estimate (FLOPs scale linearly
> in depth) or rely on the [rocprof-compute](rocprof-compute.md) roofline.

> **DeepSpeed on ROCm.** The site `pytorch` module ships a ROCm-built DeepSpeed. If
> you instead `pip install deepspeed` into a venv, its op-builder probes for a CUDA
> toolkit at import and aborts with `MissingCUDAException: CUDA_HOME does not exist`.
> Point it at ROCm (the FlopsProfiler is pure Python, needs no compiled ops):
>
> ```bash
> module load rocm
> export CUDA_HOME=${ROCM_PATH:-/opt/rocm}
> ```
>
> The **full** DeepSpeed profiler (comm logging, ZeRO timing) only applies if you
> convert training to `deepspeed.initialize` — out of scope here; use
> [torch.profiler](torch-profiler.md) for the comm breakdown instead.

## Viewing

Text only — the `FLOPS_PROFILE` line and per-module table print to stdout; no GUI.
Feed the numbers into the [rocprof-compute](rocprof-compute.md) roofline for the
achieved-vs-peak picture, which *does* have a GUI (see that page for VNC steps).

## See also

- [torch.profiler](torch-profiler.md) — measured kernel time to compare against
- [rocprof-compute](rocprof-compute.md) — achieved TFLOP/s vs. this ceiling
