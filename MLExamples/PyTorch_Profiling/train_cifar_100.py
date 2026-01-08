#!/usr/bin/env python3

"""Cifar-100 training example script, used to build profiling tools and examples for pytorch profiling on ROCm."""

__author__ = "Corey Adams, AMD"
__email__  = "corey.adams@amd.com"

import sys, os
import time

import argparse
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

# Parallelization imports:
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Dataset transforms:
import torchvision
from torchvision.transforms import v2


def init_model(args, rank):

    if args.model == "swinv2":

        from transformers import Swinv2ForImageClassification, Swinv2Config
        config = Swinv2Config(image_size=32, num_labels=100)

        model = Swinv2ForImageClassification(config)

    elif args.model == "resnet":

        # ResNet:
        from transformers import ResNetForImageClassification, ResNetConfig

        config = ResNetConfig(image_size=32, num_labels=100)
        model = ResNetForImageClassification(config)

    elif args.model == "vit":

        # ResNet:
        from transformers import ViTForImageClassification, ViTConfig

        config = ViTConfig(image_size=32, num_labels=100)
        model = ViTForImageClassification(config)

    # Move the model to the GPU and init DDP:
    model = model.to(f"cuda:{rank}")

    if args.precision == "bfloat16": model.to(torch.bfloat16)

    model=DDP(model, device_ids=[rank,])

    optimizer = torch.optim.Adam(model.parameters())

    return model, optimizer

def train(train_data, val_data, model, opt, rank):

    criterion = torch.nn.CrossEntropyLoss()

    n_steps = 0

    if args.precision == "automixed":
        precision_context = torch.autocast(device_type="cuda")
        scaler = torch.amp.GradScaler()
    else:
        import contextlib
        precision_context = contextlib.nullcontext()

    if args.torch_profile == True:
        from torch.profiler import profile, record_function, ProfilerActivity, schedule
        this_schedule = schedule(skip_first=3, wait=5, warmup=1, active=3,repeat=1)
        profiling_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, schedule=this_schedule)
    else:
        import contextlib
        profiling_context = contextlib.nullcontext()
    
    with profiling_context:

        for epoch in range(args.max_epochs):

            start = time.time()
            for i, (source, targets) in enumerate(train_data):
                
                opt.zero_grad()
                with precision_context:
                    output = model(source)
                    logits = output["logits"]

                    loss = criterion(logits, targets)

                accuracy = (torch.argmax(logits, axis=-1) == targets).to(torch.float32).mean()

                if args.precision == "automixed":
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                end = time.time()

                if args.torch_profile and rank == 0:
                    profiling_context.step()

                img_per_s = args.batch_size / (end - start)

                if rank == 0: print(f"{epoch} / {i}: loss {loss:.2f}, acc {100*accuracy:.2f}%, images / second / gpu: {img_per_s:.2f}.")
                start = time.time()
                
                n_steps += 1
                if n_steps > args.max_steps or epoch > args.max_epochs: break

            if n_steps > args.max_steps or epoch > args.max_epochs: break


        # Could add validation step here

    if args.torch_profile and rank == 0:
        profiling_context.export_chrome_trace(f"trace_{epoch}_{i}.json")
        print(profiling_context.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))

def build_dataset(args, rank, download):

    dev = torch.device(f"cuda:{rank}") if torch.cuda.is_available else torch.device("cpu")

    training_data = torchvision.datasets.CIFAR100(
        root=args.data_path,
        train=True,
        download=download,
        transform=v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(),
            v2.RandomResizedCrop(size=32, scale=[0.85,1.0], antialias=False),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    )

    val_data = torchvision.datasets.CIFAR100(
        root=args.data_path,
        train=False,
        download=download,
        transform=v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
        ])
    )

    training_data, validation_data = torch.utils.data.random_split(training_data, [0.8, 0.2], generator=torch.Generator().manual_seed(55))


    # The dataloader makes our dataset iterable 
    train_dataloader = torch.utils.data.DataLoader(training_data, 
        batch_size=args.batch_size, 
        pin_memory=True,
        shuffle=False, 
        num_workers=4,
        sampler=DistributedSampler(training_data))

    val_dataloader = torch.utils.data.DataLoader(validation_data, 
        batch_size=args.batch_size, 
        pin_memory=True,
        shuffle=False, 
        num_workers=4,
        sampler=DistributedSampler(val_data))

    # Preprocess the images:
    def preprocess(x, y):
        # CIFAR-100 is *color* images so 3 layers!
        x = x.view(-1, 3, 32, 32).to(dev)
        if args.precision == "bfloat16": x = x.to(torch.bfloat16)
        return x, y.to(dev)
    
    # Wrap the data loader to apply the preprocessing
    class WrappedDataLoader:
        def __init__(self, dl, func):
            self.dl = dl
            self.func = func
    
        def __len__(self):
            return len(self.dl)
    
        def __iter__(self):
            for b in self.dl:
                yield (self.func(*b))


    train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
    val_dataloader = WrappedDataLoader(val_dataloader, preprocess)

    return train_dataloader, val_dataloader


def init_process(backend="nccl"):


    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ

    # Detect the launch: 

    if 'NPROCS' in os.environ:
        size = int(os.environ['NPROCS'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        # Probably open mpi
        size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NPROCS' in os.environ:
        size = int(os.environ['SLURM_NPROCS'])
    else:
        size = 1

    rank = 0
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        # Probably open mpi
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_NPROCS' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])

    dist.init_process_group(backend, rank=rank, world_size=size)
    return rank

if __name__ == "__main__":


    # Create an argument parser:
    parser = argparse.ArgumentParser()

    # Data arguments:
    parser.add_argument("--data-path", "-dp", 
                        help="Top level data storage",
                        type=str,
                        default="data/")

    parser.add_argument("--batch-size", "-bs", 
                        help="Batch size per rank",
                        type=int,
                        default=256)

    parser.add_argument("--download-only", action='store_true')

    parser.add_argument("--precision", choices=["float32", "automixed", "bfloat16"],
                        default="automixed")

    parser.add_argument("--max-epochs", type=int, default=1, 
                        help="Number of epochs (maximum) to run.  Ignored if max_steps is set and is reached first.")

    parser.add_argument("--max-steps", "-ms", type=int, default=20,
                        help="Maximum number of steps to run for profiling")

    parser.add_argument("--torch-profile", action="store_true",
                        help="Activate the pytorch profiler")

    parser.add_argument("--model", type=str, 
                        choices=["resnet", "swinv2", "vit"],
                        default="resnet",
                        help="Vision classification model to use")

    args = parser.parse_args()


    # Start the process group:
    rank = init_process()

    if rank==0:
        print(args)

    download = args.download_only
    if rank != 0: download = False

    train_data, val_data = build_dataset(args, rank, download)

    if download: exit(0)

    model, opt = init_model(args, rank)

    train(train_data, val_data, model, opt, rank)

    dist.destroy_process_group()
