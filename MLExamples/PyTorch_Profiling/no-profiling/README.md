# No Profiling?

The scripts here represent bare bones launch scripts that can measure baseline performance before profiling, as reported by throughput in images / gpu / second.  You can and should execute these scripts to get a sense of total performance.  The model is doing classification of the CIFAR100 dataset, and reports training accuracy only.  Test accuracy could be added but isn't included for this performance study.

If you run on a single 300X, on ROCm 6.2, results may look something like this:

```
0 / 0: loss 5.00, acc 0.39%, images / second / gpu: 564.89.
0 / 1: loss 5.06, acc 0.00%, images / second / gpu: 11532.09.
0 / 2: loss 5.02, acc 0.00%, images / second / gpu: 13469.08.
0 / 3: loss 5.07, acc 1.95%, images / second / gpu: 13489.73.
0 / 4: loss 4.92, acc 1.95%, images / second / gpu: 4789.11.
0 / 5: loss 5.70, acc 1.17%, images / second / gpu: 10993.46.
0 / 6: loss 5.66, acc 1.17%, images / second / gpu: 12273.44.
0 / 7: loss 5.45, acc 1.95%, images / second / gpu: 12632.55.
0 / 8: loss 5.27, acc 1.56%, images / second / gpu: 12162.64.
0 / 9: loss 5.26, acc 1.95%, images / second / gpu: 12356.06.
0 / 10: loss 5.41, acc 1.17%, images / second / gpu: 12972.44.
0 / 11: loss 5.22, acc 1.56%, images / second / gpu: 12873.21.
0 / 12: loss 5.27, acc 0.39%, images / second / gpu: 13010.79.
0 / 13: loss 5.12, acc 0.39%, images / second / gpu: 12716.79.
0 / 14: loss 5.08, acc 2.34%, images / second / gpu: 12882.02.
0 / 15: loss 5.16, acc 0.39%, images / second / gpu: 12878.00.
0 / 16: loss 4.92, acc 1.17%, images / second / gpu: 12432.17.
0 / 17: loss 5.44, acc 1.56%, images / second / gpu: 12922.48.
0 / 18: loss 4.83, acc 0.78%, images / second / gpu: 12857.34.
0 / 19: loss 4.95, acc 1.95%, images / second / gpu: 13012.37.
0 / 20: loss 4.85, acc 1.56%, images / second / gpu: 12677.30.
```

On average, performance per GPU is peaking around 12000 or more images per second.  For a data paralell run, using MPI for example, performance per GPU drops slightly:

```
0 / 0: loss 4.96, acc 0.39%, images / second / gpu: 411.42.
0 / 1: loss 5.07, acc 0.78%, images / second / gpu: 9687.05.
0 / 2: loss 4.96, acc 0.39%, images / second / gpu: 11088.59.
0 / 3: loss 4.95, acc 1.17%, images / second / gpu: 10073.19.
0 / 4: loss 4.99, acc 0.78%, images / second / gpu: 3684.07.
0 / 5: loss 5.49, acc 0.78%, images / second / gpu: 6932.24.
0 / 6: loss 5.33, acc 1.95%, images / second / gpu: 8611.57.
0 / 7: loss 5.28, acc 1.95%, images / second / gpu: 8248.45.
0 / 8: loss 5.10, acc 1.17%, images / second / gpu: 8277.13.
0 / 9: loss 5.73, acc 1.95%, images / second / gpu: 8166.58.
0 / 10: loss 5.28, acc 2.34%, images / second / gpu: 8663.68.
0 / 11: loss 5.31, acc 2.34%, images / second / gpu: 8302.47.
0 / 12: loss 5.06, acc 1.56%, images / second / gpu: 8245.03.
0 / 13: loss 4.60, acc 2.73%, images / second / gpu: 8584.03.
0 / 14: loss 4.51, acc 2.73%, images / second / gpu: 10551.92.
0 / 15: loss 4.78, acc 3.12%, images / second / gpu: 10665.11.
0 / 16: loss 4.73, acc 3.52%, images / second / gpu: 10650.19.
0 / 17: loss 4.54, acc 3.91%, images / second / gpu: 10590.42.
0 / 18: loss 4.59, acc 1.95%, images / second / gpu: 10520.28.
0 / 19: loss 4.58, acc 2.94%, images / second / gpu: 3115.56.
```

Overall, this represents approximately 75% scale up efficiency for this model.  Note that this application has extremely tiny images, and the ratio of computation to communication is poor for scale up efficiency.
