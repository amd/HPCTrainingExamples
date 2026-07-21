
To run the rocprofv3 trace decoder

# ROCprof Trace Decoder

## Basic test

```
module load rocprofiler-sdk
  or
module load rocprof-tracedecoder
```

```
git clone https://github.com/amd/HPCTrainingExamples
cd HPCTrainingExamples/HIP/vectorAdd
```

```
make vectoradd
./vectoradd
rocprofv3 --att -d out -- ./vectoradd
```

Transfer the files in `out/ui*` to your local machine and read them into ROCprof Compute Viewer

Cleaning up afterwards

```
make clean
rm -rf out
```

## Matrix multiply test (DGEMM)

```
cd HPCTrainingExamples/rocprof-tracedecoder
```

```
module load rocprof-tracedecoder
```

```
make

make test
```

will run `rocprofv3 --att -d out -- ./dgemm`

Transfer files in `out/ui*` to your local system
