Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.

This training example is released under the MIT license as listed
in the top-level directory. If this example is separated from the
main directory, include the LICENSE file with it.

Major revisions by Suyash Tandon.

# (d)GEMM Application

## About
A simple (d)GEMM application created as an exercise to showcase simple matrix-matrix multiplications on AMD GPUs.
The simpler interface makes it easier to be used in training modules as opposed to other optimized and complicated gemm libraries.

## Requirements

- cmake > 2.8
- ROCm > 3.9

## Build
Follow the instructions below to configure and build the `dgemm` binary using the commands:  
```bash
mkdir build
cd build
cmake ..
make
```
To install at a specific location, provide `-DCMAKE_INSTALL_PREFIX=<path-to-install>`, for example to install in `$HOME` dir:
```bash
cmake -DCMAKE_INSTALL_PREFIX=$HOME ..
```

## Usage
Sample usage is shown below:
```bash
dgemm \
   -m 8192 \
   -n 8192 \
   -k 8192 \
   -i 10 \
   -r 10 \
   -d DEVICE_LIST \
   -o $(hostname)_dgemm.json
```

where,
   - `m` is row count of matrix A
   - `n` is column count of matrix A
   - `k` is column count of matrix B
   - `i` is iteration count to perform
   - `r` is number of repetitions of dgemm to perform when evaluating
     flops
   - `d` is a comma separated list of devices to use, indexed at zero (e.g. 0,1,...)
   - `o` to give filename to write all data. If not `.csv`, will write in `.json` (optional)

## Output

GEMM operations on each gpu is ran asynchronously, and the data
printed to `stdout` aims only to show the progress for each
gpu:
```bash
./dgemm -m 8192 -n 8192 -k 8192 -i 3 -r 10 -d 0,1,2,3 -o dgemm.csv
2     1   27.56
0     1   27.63
3     1   27.56
1     1   27.63
2     2   27.56
0     2   27.63
3     2   27.56
1     2   27.63
2     3   27.56
0     3   27.63
3     3   27.56
1     3   27.63
```
Summary of the results are dumped to `stdout` at the end of the run:
```bash
  DEV |        MIN |        MAX |    AVERAGE |    STD Dev
-----------------------------------------------------------
    0 |    27.6259 |    27.6259 |    27.6259 |          0
    1 |    27.6259 |    27.6259 |    27.6259 |          0
    2 |    27.5567 |    27.5567 |    27.5567 |          0
    3 |    27.5567 |    27.5567 |    27.5567 |          0
```

If an output file is specified, the complete data for each iteration
including timestamps are printed to file in either `json` or `csv`
format, depending on the specific output file extension via `-o`.


### json

Default output format is json unless `csv` extension is specified in
output filename. The results per iteration are recorded along with the
local timestamp when the flop-rate was estimated. Output format is

```text
{
  "flop_rates": {
     "<device_id_1>": [<flop_rates>, ...],
     "<device_id_2>": [<flop_rates>, ...],
     ...
  },
  "times": {
     "<device_id_1>": ["<timestamp>", ...],
     "<device_id_2>": ["<timestamp>", ...],
     ...
  },
  "args": {
     <input args as key-value pairs>
  }
}
```

where `<timestamp>` format is `YYYY-mm-dd HH:MM:SS.ZZZ`, and
`<device_id_N>` are specified input device ids (0, 1, 2, ...).


### csv

If the output file name has `csv` as extension, data is written in
comma-saparated format. There is a single header line, followed by
data, where the header for this case is as follows

```text
t_<N>,flops_<N>,[...]
```

where `N` is an integer representing the device id. Device ids are not
guaranteed to be ordered.

Example `csv` output:

```bash
$ cat dgemm.csv
t_3,flops_3,t_2,flops_2,t_0,flops_0,t_1,flops_1
"2022-12-08 15:17:15.232",27.556682,"2022-12-08 15:17:15.087",27.556682,"2022-12-08 15:17:15.228",27.625920,"2022-12-08 15:17:15.437",27.625920
"2022-12-08 15:17:15.631",27.556682,"2022-12-08 15:17:15.486",27.556682,"2022-12-08 15:17:15.627",27.625920,"2022-12-08 15:17:15.836",27.625920
"2022-12-08 15:17:16.030",27.556682,"2022-12-08 15:17:15.885",27.556682,"2022-12-08 15:17:16.025",27.625920,"2022-12-08 15:17:16.234",27.625920
```
