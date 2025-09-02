#!/bin/bash
time apptainer pull rocm-pytorch.sif docker://rocm/pytorch:latest
sbatch pytorch_mnist_apptainer.batch
