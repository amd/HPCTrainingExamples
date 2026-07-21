import os
import sys
import torch
import torch.distributed as dist
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

def rccl_bandwidth_test_func(config):
    size_mb = config.get("size_mb", 1024)
    iterations = config.get("iterations", 10)
    world_size = dist.get_world_size()
    
    n_elements = (size_mb * 1024 * 1024) // 4
    tensor = torch.ones(n_elements, device="cuda")
    
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_event.record()
    
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_s = (total_time_ms / 1000.0) / iterations
    
    size_bytes = size_mb * 1024 * 1024
    bus_bandwidth_gbps = (size_bytes * 2 * (world_size - 1) / world_size) / avg_time_s * 8 / 1e9

    ray.train.report({"gbps": round(bus_bandwidth_gbps, 2), "status": "READY"})

def main():
    if len(sys.argv) < 4:
        print("Usage: python ray_health_rccl.py <nnodes> <cpus_per_node> <gpus_per_node>")
        sys.exit(1)

    nnodes = int(sys.argv[1])
    cpus_per_node = int(sys.argv[2])
    gpus_per_node = int(sys.argv[3])
    
    total_workers = nnodes * gpus_per_node
    cpus_per_worker = cpus_per_node // gpus_per_node

    ray.init(address="auto", ignore_reinit_error=True)
    
    trainer = TorchTrainer(
        train_loop_per_worker=rccl_bandwidth_test_func,
        train_loop_config={"size_mb": 1024, "iterations": 10},
        scaling_config=ScalingConfig(
            num_workers=total_workers, 
            use_gpu=True,
            resources_per_worker={"CPU": cpus_per_worker, "GPU": 1}
        )
    )

    print(f"Testing {total_workers} workers across {nnodes} nodes...")
    if nnodes > 1:
        print(f"Running multinode, slingshot fabric bandwidth should be around 400 gbps")
    else:
        print(f"Running single node, infinity fabric bandwidth should be around 1600 gbps")
    result = trainer.fit()
    if result and result.metrics:
        print(f"RCCL Final Bandwidth: {result.metrics.get('gbps', 'N/A')} Gbps")
    else:
        print("RCCL health check completed, but summary metrics were not captured.")

if __name__ == "__main__":
    main()
