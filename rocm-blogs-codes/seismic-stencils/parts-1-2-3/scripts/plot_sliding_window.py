import helper
import matplotlib.pyplot as plt
import numpy as np


d = np.loadtxt("figures/sliding_window.csv", skiprows=1, delimiter=",")
nw = d[:,0]
speedup = d[:,1]
pop = d[:,2]
rw = d[:,3]

plt.subplot(1,2,1)
plt.semilogx(nw, speedup, label="Achieved Speedup")
plt.semilogx(nw, (1 + 9) / (1 + (8 + nw) / nw), label="Predicted Speedup")
plt.legend()
plt.xlabel("Window dimension $n_w$")
plt.ylabel("Speedup")
plt.subplot(1,2,2)
plt.semilogx(nw, rw, label="Achieved R/W")
plt.semilogx(nw, (8 + nw) / nw, label="Predicted R/W")
plt.legend()
plt.xlabel("Window dimension $n_w$")
plt.ylabel("Read/Write Ratio")
plt.savefig("../docs/figures/sliding_window.png")
#plt.show()

plt.clf()
plt.semilogx(nw, pop)
plt.xlabel("Window dimension $n_w$")
plt.ylabel("Bandwidth (PoP)")
plt.ylim([0, 100])
print(max(speedup))
print(min(pop))
print(max(pop))
plt.savefig("../docs/figures/sliding_window_bandwidth.png")
