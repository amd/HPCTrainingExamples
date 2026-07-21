To run the microbenchmark for an op:
```
python torch_tensor_ops_bench.py --op <op_name>
```

The script also takes optional arguments:
```
--dtype [=fp32 | fp16 | bf16]
--device [=cuda | cpu]
--input-dim dims separated by '-', default "64-1024-1024"
--op-type [=None | binary(for a binary op)]
