

import tomli
import helper

fd_window_z_bw = []
fd_z_bw = []

fd_window_z = []
fd_z = []
nws = []

for nw in range(0, 1000, 10):
    if nw == 0:
        nw = 1
    with open(f"logs/1000x1000x1000/baseline_0_align_1_nw_{nw}/results.toml", "rb") as t:
        tdict = tomli.load(t)
    curr_fd_window_z = helper.get_metrics(tdict["compute_fd_sliding_window_z_gpu_kernel"])
    fd_window_z.append(curr_fd_window_z)
    curr_fd_z = helper.get_metrics(tdict["compute_fd_z_gpu_kernel"])
    fd_z.append(curr_fd_z)

    fd_window_z_bw.append(curr_fd_window_z["HBM Bandwidth"].mean)
    fd_z_bw.append(curr_fd_z["HBM Bandwidth"].mean)
    nws.append(nw)


for nw, bw_window, bw in zip(nws, fd_window_z_bw, fd_z_bw):
    print(nw, bw_window, bw)

