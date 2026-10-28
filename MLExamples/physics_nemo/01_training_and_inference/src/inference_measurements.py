"""Size and synchronized forward-pass measurements for the workshop."""
from pathlib import Path
import statistics
import time


def model_sizes(model, checkpoint_dir):
    """Decimal GB: model archive on disk and parameters/buffers in memory."""
    archives = list(Path(checkpoint_dir).glob("*.mdlus"))
    if not archives:
        raise FileNotFoundError("No model archive found in " + str(checkpoint_dir))
    return {
        "model_archive_gb": sum(p.stat().st_size for p in archives) / 1e9,
        "model_tensor_gb": sum(t.numel() * t.element_size()
                               for t in list(model.parameters()) + list(model.buffers())) / 1e9,
    }


def measure_inference(model, sample, data_stats, device, warmup=3, repeats=10):
    """Time one full trajectory per forward pass; inputs already on device.

    Excludes loading, preprocessing, transfers, and error computation. GPU
    synchronization includes completion of asynchronous CUDA/HIP work.
    """
    import torch

    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    durations = []
    with torch.no_grad():
        for _ in range(warmup):
            prediction = model(sample=sample, data_stats=data_stats)
        synchronize()
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            prediction = model(sample=sample, data_stats=data_stats)
            synchronize()
            durations.append(time.perf_counter() - start)
    median = statistics.median(durations)
    return prediction, {
        "median_inference_ms": median * 1000,
        "min_inference_ms": min(durations) * 1000,
        "max_inference_ms": max(durations) * 1000,
        "trajectories_per_second": 1 / median,
        "warmup": warmup,
        "repeats": repeats,
        "durations_seconds": durations,
    }
