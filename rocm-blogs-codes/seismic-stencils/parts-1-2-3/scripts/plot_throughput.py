import helper
import matplotlib.pyplot as plt
import numpy as np


d = np.loadtxt("figures/throughput.csv", skiprows=1, delimiter=",")
n = d[:,0]
fd_z = d[:,1]
afd_z = d[:,2]
window_fd_z = d[:,3]
awindow_fd_z = d[:,4]

plt.plot(n, fd_z, label="Baseline R=4 <64x8> (Unaligned)")
plt.plot(n, afd_z, label="Baseline R=4 <64x8> (Aligned)")
plt.legend()
plt.xlabel("Grid size ($n$)")
plt.ylabel("Throughput (Gcell/s)")
plt.ylim([0, 150])
plt.savefig("../docs/figures/throughput_baseline.png", res=300)

plt.plot(n, window_fd_z, label="sliding window <R=4> (Unaligned)")
plt.plot(n, awindow_fd_z, label="sliding window <R=4> (Aligned)")
plt.legend()
plt.xlabel("Grid size ($n$)")
plt.ylabel("Throughput (Gcell/s)")
plt.ylim([0, 150])
plt.savefig("../docs/figures/throughput_sliding_window.png")

plt.clf()
plt.plot(n, fd_z, label="Baseline <64x8> R=4")
plt.plot(n, awindow_fd_z, label="Optimized <64x8> R=4")
plt.legend()
plt.xlabel("Grid size ($n$)")
plt.ylabel("Throughput (Gcell/s)")
plt.ylim([0, 150])
plt.savefig("../docs/figures/throughput.png", res=300)
plt.show()
