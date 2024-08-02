import helper
import matplotlib.pyplot as plt
import numpy as np


d = np.loadtxt("../figures/new_sliding_window.csv", skiprows=1, delimiter=",")
nw = d[:,0]
effbw = d[:,1]
read = d[:,2]
write = d[:,3]
achbw = d[:,4]
ratio = np.divide(read,write)

plt.subplot(1,2,1)
plt.semilogx(nw, effbw, label="Effective bandwidth")
plt.semilogx(nw, achbw, label="Achieved bandwidth")
#plt.semilogx(nw, (1 + 9) / (1 + (8 + nw) / nw), label="Predicted Speedup")
plt.legend()
plt.xlabel("Window dimension $n_w$")
plt.ylabel("Memory bandwidth (GB/s)")
plt.subplot(1,2,2)
plt.semilogx(nw, ratio, label="Achieved R/W")
plt.semilogx(nw, (8 + nw) / nw, label="Predicted R/W")
plt.legend()
plt.xlabel("Window dimension $n_w$")
plt.ylabel("Read/Write Ratio")
plt.savefig("../../docs/figures/new_sliding_window.png")
#plt.show()

#plt.clf()
#plt.semilogx(nw, pop)
#plt.xlabel("Window dimension $n_w$")
#plt.ylabel("Bandwidth (PoP)")
#plt.ylim([0, 100])
#print(max(speedup))
#print(min(pop))
#print(max(pop))
#plt.savefig("../docs/figures/sliding_window_bandwidth.png")
