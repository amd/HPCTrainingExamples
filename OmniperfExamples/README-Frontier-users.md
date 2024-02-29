### Instructions for loading Omniperf on Frontier:

To load the appropriate environment, you should be able to simply run:
```
module use /autofs/nccs-svm1_sw/crusher/amdsw/modules
module load omniperf/1.1.0-PR1
```
This should pull in Omniperf, ROCm, and all the dependencies necessary to run our exercises.
It is worthy of note that this version of Omniperf is a pre-release candidate for 1.1.0.

>Note: By default this loads ROCm 5.3.0, which may not show the issue talked about in Exercise 3: Register Occupancy Limiter

To allocate an interactive job on Frontier:
```
salloc -N 1 -p batch --reservation=hip_training_2023_10_16 --gpus=1 -t 10:00 -A <project>
```

Use your project ID in the project field. If you're unsure of what projects are available to you, run the above command without the `-A` option, and it will report a list of your valid projects.

Outside our reservation window, you can do:
```
salloc -N 1 -p batch --gpus=1 -t 10:00 -A <project>
```


