#!/usr/bin/env python3

import os
import tqdm
import torch
import random
import argparse
import neuralop
import numpy as np
import matplotlib.pyplot as plt

import the_well
import the_well.utils
import the_well.utils.download


if torch.cuda.is_available:
    device = "cuda" 
    print("Running on a GPU!")
else:
    device = "cpu"
    print("Running on the CPU!")


parser = argparse.ArgumentParser(
        prog='FNO training script'
        )
parser.add_argument(
        '--batchsize',
        type=int,
        default=4,
        help='Batchsize for training'
        )
parser.add_argument(
        '--num_episodes',
        type=int,
        default=5,
        help='Number of episodes the training is run for'
        )
parser.add_argument(
        '--n_steps_input',
        default=4,
        type=int,
        help='Number of timesteps used as input for model'
        )
parser.add_argument(
        '--dataset_name',
        type=str,
        default='turbulent_radiative_layer_2D',
        help='Name of "The Well" dataset to use'
        )
parser.add_argument(
        '--base_path',
        type=str,
        default='./',
        help='Path to base directory containing the "datasets/" folder with the datasets'
        )
parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Manual seed for reproducible results'
        )
parser.add_argument(
        '--torch_profile',
        action='store_true',
        help='Enable the PyTorch profiler'
        )
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.torch_profile == True:
    from torch.profiler import profile, record_function, ProfilerActivity, schedule
    this_schedule = schedule(skip_first=3, wait=5, warmup=1, active=3,repeat=1)
    profiling_context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, schedule=this_schedule)
else:
    import contextlib
    profiling_context = contextlib.nullcontext()


def get_datasets(
        dataset_name,
        base_path = "./",
        n_steps_input = 1
        ):
    """Downloads and prepares a dataset from The Well"""

    # Download datasets only if not already present
    dataset_path = os.path.join(base_path, 'datasets/', dataset_name)
    if not os.path.isdir(dataset_path):
        print(f'Downloading dataset "{dataset_name}" to location {dataset_path}!')
        the_well.utils.download.well_download(
                base_path=base_path,
                dataset=dataset_name,
                split="train"
                )
        the_well.utils.download.well_download(
                base_path=base_path,
                dataset=dataset_name,
                split="valid"
                )
    else:
        print(f'Found dataset at location {dataset_path}!')

    dataset_train = the_well.data.WellDataset(
            well_base_path = f"{base_path}/datasets",
            well_dataset_name = dataset_name,
            well_split_name = "train",
            n_steps_input = n_steps_input,
            n_steps_output = 1
            )
    dataset_valid = the_well.data.WellDataset(
            well_base_path = f"{base_path}/datasets",
            well_dataset_name = dataset_name,
            well_split_name = "valid",
            n_steps_input = n_steps_input,
            n_steps_output = 1,
            full_trajectory_mode = True,
            use_normalization = False,
            )
    
    return dataset_train, dataset_valid, dataset_train.metadata.n_fields


def main():
    # Load datasets from "The Well" (download if necessary)
    dataset_train, dataset_valid, n_fields = get_datasets(
            dataset_name = args.dataset_name,
            base_path = args.base_path,
            n_steps_input = args.n_steps_input
            )

    # Instatiate FNO model with same architecture as in reference paper
    model = neuralop.models.FNO(
        n_modes = (16, 16),
        in_channels = args.n_steps_input * n_fields,
        out_channels = n_fields,
        hidden_channels = 128,
        n_layers = 5,
        ).to(device)
    
    # Use standard Adam with optimal LR from paper
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-3
            )
    
    # Use simple MSE loss for training
    loss = torch.nn.MSELoss()
    
    # Create Data Loader
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset_train,
        shuffle = True,
        batch_size = args.batchsize,
##> Add for improvement
#        num_workers = 2,
#        prefetch_factor=4,
#        pin_memory = True
##> Add for improvement
    )
    
    with profiling_context:
        # Training Loop
        for epoch in range(args.num_episodes):
            for i, batch in enumerate(tqdm.tqdm(data_loader, unit="batches")):
                if args.torch_profile:
                    profiling_context.step()

                # Nullify the gradients for batch
                optimizer.zero_grad()

                # Get input and output fields from batch and transform from
                # (Batch, Timesteps, x, y, Fields), -> (Batch, (Timesteps*Fields), x, y)
                x = batch["input_fields"].to(device)
                x = torch.permute(x, (0, 1, 4, 2, 3))
                x = torch.flatten(x, start_dim=1, end_dim=2)

                y = batch["output_fields"].to(device)
                y = torch.permute(y, (0, 1, 4, 2, 3))
                y = torch.flatten(y, start_dim=1, end_dim=2)
        
                y_model = model(x)
        
                output = loss(y_model, y)
                output.backward()
                optimizer.step()
        
    if args.torch_profile:
        profiling_context.export_chrome_trace(f"trace_{epoch}_{i}.json")
        print(profiling_context.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))


    # Run validation
    item = dataset_valid[1]
    x = item["input_fields"].to(device)
    y = item["output_fields"].to(device)
    
    # Get whole trajectory by stacking inputs and outputs
    traj = torch.cat( (x, y), dim=0)
    traj_model = traj.clone()

    # Evaluate the model auto-regressivly on the whole trajectory
    with torch.no_grad():
        for i in range(args.n_steps_input, (traj.shape)[0]-1):
            x_in = traj_model[i-args.n_steps_input:i]
            x_in = torch.reshape(x_in, (1,) + x_in.size())
            x_in = torch.permute(x_in, (0, 1, 4, 2, 3))
            x_in = torch.flatten(x_in, start_dim=1, end_dim=2)

            y_model = model(x_in)

            # Append prediction to feed back into model
            traj_model[i+1] = torch.permute(y_model, (0, 2, 3, 1) )

    traj = traj.cpu()
    traj_model = traj_model.cpu()
    
    # Visu auto-regresive predictions by model
    n_steps_visu = 5  # Total number of steps to show
    n_spacing = 2     # Step increment between two steps
    fig, axs = plt.subplots(n_steps_visu, 2, figsize=(3, 0.65*n_steps_visu))
    for t in range(n_steps_visu):
        idx = args.n_steps_input - 1 + t * n_spacing
        axs[t, 0].imshow(traj[idx,:,:,0])
        axs[t, 0].set_xticks([])
        axs[t, 0].set_yticks([])
        axs[t, 0].set_ylabel(f"$t={t * n_spacing}$")

        axs[t, 1].imshow(traj_model[idx,:,:,0])
        axs[t, 1].set_xticks([])
        axs[t, 1].set_yticks([])

    axs[0, 0].set_title("Reference")
    axs[0, 1].set_title("Predictions")
    
    plt.tight_layout()
    plt.savefig("model_prediction.png")


if __name__ == "__main__":
   main() 
