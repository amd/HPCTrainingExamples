\pagebreak{}

# Rocprof

**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

We discuss an example on how to use the tools from `rocprof`.

## Initial Setup

First, setup the environment:

```bash
salloc --cpus-per-task=8 --mem=0 --ntasks-per-node=4 --gpus=1
module load rocm
```

Download the examples repo and navigate to the `HIPIFY` exercises:

```bash
cd ~/HPCTrainingExamples/HIPIFY/mini-nbody/hip/
```

Update the bash scripts with `$ROCM_PATH`:

```bash
sed -i 's/\/opt\/rocm/${ROCM_PATH}/g' *.sh
```

Compile and run the `nbody-orig.hip` program (the script below will do both, for several values of `nBodies`):

```bash
./HIP-nbody-orig.sh
```

To compile explicitly without `make` you can do (considering for example `nbody-orig`):

```bash
hipcc -I../ -DSHMOO nbody-orig.hip -o nbody-orig
```

And then run with:

```bash
./nbody-orig <nBodies>
```

The procedure for compiling and running a single example applies to the other programs in the directory. The default value for `nBodies` is 30000 for all the examples.

##  Run ROCprof and Inspect the Output

Run `rocprof` to obtain the hotspots list (considering for example `nbody-orig`):

```bash
rocprof --stats --basenames on nbody-orig 65536
```

In the above command, the `--basenames on` flag removes the kernel arguments from the output, for ease of reading. Throughout this example, we will always use 65536 as a value for `nBodies`, since `nBodies` is used to define the number of work groups in the thread grid:

```bash
nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE
``` 

Check `results.csv` to find, for each invocation of each kernel, details such as grid size (`grd`), workgroup size (`wgr`), LDS used (`lds`), scratch used if register spilling happened (`scr`), number of SGPRs and VGPRs used, etc. Note that grid size is equal to the total number of ***work-items (threads)***, not the number of work groups. This is the output that is useful if you allocate shared memory dynamically, for instance.

Additionally, you can check the statistics result file called `results.stats.csv`, displayed one line per kernel, sorted in descending order of durations.

You can trace HIP, GPU and Copy activity with `--hip-trace`: 

```bash
rocprof --hip-trace nbody-orig 65536
```

The output is the file `results.hip_stats.csv`, which lists the HIP API calls and their durations, sorted in descending order. This can be useful to find HIP API calls that may be bottlenecks.

You can also profile the HSA API by adding the `--hsa-trace` option. This is useful if you are profiling OpenMP target offload code, for instance, as the compiler implements all GPU offloading via the HSA layer:

```bash
rocprof --hip-trace --hsa-trace nbody-orig 65536
```

In addition to`results.hip_stats.csv`, the command above will create the file `results.hsa_stats.csv` which contains the statistics information for HSA calls.

## Visualization with Perfetto

The `results.json` JSON file produced by `rocprof` can be downloaded to your local machine and viewed in Perfetto UI. This file contains the timeline trace for this application, but shows only GPU, Copy and HIP API activity. 

Once you have downloaded the file, open a browser and go to [https://ui.perfetto.dev/](https://ui.perfetto.dev/).
Click on `Open trace file` in the top left corner.
Navigate to the `results.json` you just downloaded.
Use WASD to navigate the GUI

![image](https://user-images.githubusercontent.com/109979778/225451481-46ffd521-2453-4caa-8d28-fa4e0f4c4889.png)

To read about the GPU hardware counters available, inspect the output of the following command:

```bash
less $ROCM_PATH/lib/rocprofiler/gfx_metrics.xml
```

In the output displayed, look for the section associated with the hardware on which you are running (for instance gfx90a).

Create a `rocprof_counters.txt` file with the counters you would like to collect, for instance:

```bash
touch rocprof_counters.txt
```

and write this in `rocprof_counters.txt` as an example:

```bash
pmc : Wavefronts VALUInsts
pmc : SALUInsts SFetchInsts GDSInsts
pmc : MemUnitBusy ALUStalledByLDS
```

Execute with the counters we just added, including the `timestamp on` option which turns on GPU kernel timestamps:

```bash
rocprof --timestamp on -i rocprof_counters.txt  nbody-orig 65536
```

You'll notice that `rocprof` runs 3 passes, one for each set of counters we have in that file.

View the contents of `rocprof_counters.csv` for the collected counter values for each invocation of each kernel:

```bash
cat rocprof_counters.csv
