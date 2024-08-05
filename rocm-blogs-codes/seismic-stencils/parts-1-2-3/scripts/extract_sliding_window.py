import tomli
import helper

nx=1000
ny=1000
nz=1000
nws=list(range(1,10,1))
nws+=list(range(10,20,2))
nws+=list(range(20,40,4))
nws+=list(range(40,80,8))
nws+=list(range(80,200,16))
nws+=list(range(200,1032,32))
nws+=[500]
nws.sort()
r=4
# With proper padding
align=64
use_offset=1
size = nx * ny * nz
ideal = 4 * size / 1e6
baseline_kernel = "compute_fd_z_gpu_kernel"
kernel = "compute_fd_sliding_window_z_gpu_kernel"

fn = lambda nw: helper.filename("logs/sliding_window_2023_04_28", nx, ny, nz, "vary_nw",
        r, align, nw, use_offset)

filenames = []
for nw in nws:
    filenames.append(fn(nw))

dur_baseline = helper.read_kernel_seq(filenames, baseline_kernel, "Duration")[0]
dur = helper.read_kernel_seq(filenames, kernel, "Duration")
print(min(dur))
hbmr = helper.read_kernel_seq(filenames, kernel, "HBM Read")
hbmw = helper.read_kernel_seq(filenames, kernel, "HBM Write")
bw = helper.read_kernel_seq(filenames, kernel, "HBM Bandwidth")

helper.print_csv("$n_w$", "Speedup", "Bandwidth (PoP)", "Read-Write Ratio")
for nw, duri, bwi, r, w in zip(nws, dur, bw, hbmr, hbmw):
    helper.print_csv(nw, dur_baseline / duri, helper.bwpop(bwi), r / w)
