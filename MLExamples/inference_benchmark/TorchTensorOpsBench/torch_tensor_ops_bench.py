import torch
import torch.nn as nn

import numpy as np
import time
import argparse
import sys

dtype_map = {"fp32" : torch.float32,
             "fp16" : torch.float16,
             "bf16" : torch.bfloat16}

# important sizes
# BERT, DLRM
# list of ops
# [(op_name, op_type, inp1_dim, inp2_dim)]
model_ops = [
    ('add', 'binary', '32-128-1024', '32-128-1024'),
    ('add', 'binary', '32-16-128-128','32-1-1-128'),
    ('add', 'binary', '64-128-1024', '64-128-1024'),
    ('add', 'binary', '64-16-128-128', '64-1-1-128'),
    ('add', 'binary', '4-512-1024', '4-512-1024'),
    ('add', 'binary', '4-16-512-512', '4-1-1-512'),
    ('add', 'binary', '8-512-1024', '8-512-1024'),
    ('add', 'binary', '8-16-512-512', '8-1-1-512'),
    ('add_', 'binary', '32-128-1024', '1024'),
    ('add_', 'binary', '64-128-1024', '1024'),
    ('add_', 'binary', '4-512-1024', '1024'),
    ('add_', 'binary', '8-512-1024', '1024'),
    ('add_', 'binary', '512-13', '512-13'),
    ('add_', 'binary', '512-512', '512-512'),
    ('add_', 'binary', '256-512', '256-512'),
    ('add_', 'binary', '256-256', '256-256'),
    ('add_', 'binary', '128-256', '128-256'),
    ('add_', 'binary', '128-128', '128-128'),
    ('add_', 'binary', '1024-480', '1024-480'),
    ('add_', 'binary', '1024-1024', '1024-1024'),
    ('add_', 'binary', '512-1024', '512-1024'),
    ('add_', 'binary', '1-256', '1-256'),
    ('add_', 'binary', '1', '1'),
    ('div', 'binary', '32-16-128-128', '1'),
    ('div', 'binary', '64-16-128-128', '1'),
    ('div', 'binary', '4-16-512-512', '1'),
    ('div', 'binary', '8-16-512-512', '1'),
    ('sum', 'reduction', '32-128-1024'),
    ('sum', 'reduction', '64-128-1024'),
    ('sum', 'reduction', '4-512-1024'),
    ('sum', 'reduction', '8-512-1024'),
    ('add_', 'sparse', '32709138-128', '851968'),
    ('relu_', 'unary', '32768-512'),
    ('relu_', 'unary', '32768-256'),
    ('relu_', 'unary', '32768-128'),
    ('relu_', 'unary', '32768-1024'),
]

# initial set of ops.
# TODO: add more ops to this list
binary_ops = ['add', 'mul', 'div', 'sub', 'eq']
unary_ops = ['exp', 'relu', 'tanh', 'sqrt']
reduction_ops = ['sum', 'prod', 'norm', 'max', 'mean', 'std', 'var', 'argmax', 'argmin']
predefined_ops = binary_ops + unary_ops + reduction_ops

def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()

def benchmark(op_str, args):
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]
    input1_dim = [int(dim) for dim in args.input1_dim.split("-")]
    input1 = torch.randn(input1_dim, device=device, dtype=dtype)
    sparse_str = "(sparse)" if args.op_type == "sparse" else ""
    dtype_str = "(" + args.dtype + ")" if args.append_dtype else ""
    op_meta = op_str + sparse_str + dtype_str + "(" + args.input1_dim
    op_args = []

    if args.op_type == 'sparse':
        last_dim = input1_dim[-1]
        num_indices = int(args.input2_dim)
        indices = torch.cuda.LongTensor(np.random.uniform(0, input1_dim[0], [1, num_indices]))
        values = torch.cuda.FloatTensor(np.random.uniform(-1, 1, [num_indices, last_dim]))
        values = values.to(dtype=dtype)
        input2 = torch.cuda.sparse.FloatTensor(indices, values, size=input1_dim)
        op_args.append(input2)
        op_meta += "," + str(num_indices) + "-" + str(last_dim)

    if op_str in binary_ops or args.op_type == 'binary':
        assert args.input2_dim, "input2_dim should be set for binary op - {}".format(op_str)
        input2_dim = [int(dim) for dim in args.input2_dim.split("-")]
        input2 = torch.randn(input2_dim, device=device, dtype=dtype)
        op_meta += "," + args.input2_dim
        op_args.append(input2)

    op_meta += ")"
    args.op_meta = op_meta
    if args.inplace and ((op_str in binary_ops) or (op_str in unary_ops)) and op_str[-1] != '_':
        op_str += '_'  #inplace

    op = getattr(input1, op_str)

    try:
        # warmup iterations
        for _ in range(args.num_warmup_iters):
            op(*op_args)

        # main iterations
        with torch.autograd.profiler.profile(enabled=args.enable_profiling, use_cuda=args.use_gpu) as prof:
            start_time = time_wrap(args.use_gpu)
            for _ in range(args.num_iters):
                op(*op_args)
            end_time = time_wrap(args.use_gpu)
            time_per_iter = 1000.0*(end_time - start_time)/args.num_iters
            print("{:45} : {:.2f} ms/it".format(op_meta, time_per_iter))

        if args.enable_profiling:
            if args.use_gpu:
                print(prof.key_averages().table(sort_by="cuda_time_total"))
            else:
                print(prof.key_averages().table(sort_by="cpu_time_total"))
    except RuntimeError as e:
        raise RuntimeError("{} operator failed with error: {}".format(op_meta, str(e)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', required=False, type=str)
    parser.add_argument("--dtype", default='fp32', required=False, type=str, choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument("--input1-dim", default="64-1024-1024", type=str, required=False)
    parser.add_argument("--input2-dim", default="64-1024-1024", type=str, required=False)
    parser.add_argument("--op", default='add', required=False, type=str)
    parser.add_argument("--op-type", default=None, required=False, type=str)
    parser.add_argument("--inplace", default=False, action="store_true")
    parser.add_argument("--run-predefined", default=False, action="store_true")
    parser.add_argument("--run-model-ops", default=False, action="store_true")
    parser.add_argument("--num-iters", default=20, type=int, required=False)
    parser.add_argument("--num-warmup-iters", default=5, type=int, required=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--append-dtype", action="store_true", default=False)

    args = parser.parse_args()
    args.use_gpu = True if 'cuda' in args.device else False

    print("========= Milliseconds per iteration for PyTorch operators with {} dtype =========".format(args.dtype))
    if args.run_predefined:
        for op_str in predefined_ops:
            benchmark(op_str, args)
    elif args.run_model_ops:
        assert (not args.inplace), "inplace should not be set when running model ops"
        for op_info in model_ops:
            if len(op_info) == 4:
                op_str, args.op_type, args.input1_dim, args.input2_dim = op_info
            else:
                op_str, args.op_type, args.input1_dim = op_info
                args.input2_dim = None
            benchmark(op_str, args)
    else:
        benchmark(args.op, args)
