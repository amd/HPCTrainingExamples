import sys
import torch
import torch.nn as nn
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import socket

def health_check_func(config):
    model = nn.Linear(10, 10).cuda()
    model = ray.train.torch.prepare_model(model)
    
    input_data = torch.randn(32, 10).cuda()
    loss = model(input_data).sum()
    loss.backward()

    ray.train.report({
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "status": "SUCCESS"
    })

def main():
    if len(sys.argv) < 4:
        print("Usage: python ray_health_overall.py <nnodes> <cpus_per_node> <gpus_per_node>")
        sys.exit(1)

    nnodes = int(sys.argv[1])
    cpus_per_node = int(sys.argv[2])
    gpus_per_node = int(sys.argv[3])
    
    total_workers = nnodes * gpus_per_node
    cpus_per_worker = cpus_per_node // gpus_per_node

    ray.init(address="auto")

    trainer = TorchTrainer(
        train_loop_per_worker=health_check_func,
        scaling_config=ScalingConfig(
            num_workers=total_workers,
            use_gpu=True,
            resources_per_worker={"CPU": cpus_per_worker, "GPU": 1}
        )
    )

    print(f"Verifying {total_workers} workers on {nnodes} nodes...")
    result = trainer.fit()
    print("RCCL Cluster Health Verified.")

if __name__ == "__main__":
    main()
