import tomli
import helper

nws = [1, 10, 100, 1000]
nx=1000
ny=1000
nz=1000
r=4
# With proper padding
align=64
use_offset=1
peak=1638.4
size = nx * ny * nz

fn = lambda nw, align, use_offset: helper.filename("logs/baseline_2023_04_26", nx, ny, nz, "baseline", r, align, nw ,use_offset)



# Unaligned data
filenames = []
for nw in nws:
    filenames.append(fn(nw, 1, 0))

fd_z_dur = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "Duration")
fd_z_bw = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "HBM Bandwidth")
fd_z_read = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "HBM Read")
fd_z_write = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "HBM Write")

fd_window_z_dur = helper.read_kernel_seq(filenames, "compute_fd_sliding_window_z_gpu_kernel","Duration")
fd_window_z_bw = helper.read_kernel_seq(filenames, "compute_fd_sliding_window_z_gpu_kernel", "HBM Bandwidth")
fd_window_z_read = helper.read_kernel_seq(filenames,  "compute_fd_sliding_window_z_gpu_kernel", "HBM Read")
fd_window_z_write = helper.read_kernel_seq(filenames, "compute_fd_sliding_window_z_gpu_kernel", "HBM Write")

# Aligned data
filenames = []
for nw in nws:
    filenames.append(fn(nw, align, 1))

afd_z_dur = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "Duration")
afd_z_bw = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "HBM Bandwidth")
afd_z_read = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "HBM Read")
afd_z_write = helper.read_kernel_seq(filenames, "compute_fd_z_gpu_kernel", "HBM Write")

afd_window_z_dur = helper.read_kernel_seq(filenames, "compute_fd_sliding_window_z_gpu_kernel","Duration")
afd_window_z_bw = helper.read_kernel_seq(filenames, "compute_fd_sliding_window_z_gpu_kernel", "HBM Bandwidth")
afd_window_z_read = helper.read_kernel_seq(filenames,  "compute_fd_sliding_window_z_gpu_kernel", "HBM Read")
afd_window_z_write = helper.read_kernel_seq(filenames, "compute_fd_sliding_window_z_gpu_kernel", "HBM Write")



print("Throughput Gcells/s")
helper.print_csv("Window Dim. Z", "Windowed", "FD No window")
for nw, durw, dur in zip(nws, fd_window_z_dur, fd_z_dur):
    helper.print_csv(nw, helper.gcells(durw, size), helper.gcells(dur, size))
print("")

print("Bandwidth (PoP)")
helper.print_csv("nw", "fdw_z", "fd_z")
for nw, bw_window, bw in zip(nws, fd_window_z_bw, fd_z_bw):
    helper.print_csv(nw, bw_window / peak * 100.0, bw / peak * 100.0)
print("")

print("Read/Write ratio")
helper.print_csv("nw", "fdw_z", "fd_z")
for nw, rw, ww, r, w in zip(nws, fd_window_z_read, fd_window_z_write, fd_z_read, fd_z_write):

    ratio_w = rw / ww
    ratio = r / w
    helper.print_csv(nw, ratio_w, ratio)
print("")

print("Speedup")
baseline_avg = sum(fd_z_dur) / len(fd_z_dur)
helper.print_markdown_header("n_w", "Speedup")
for nw, durw in zip(nws, fd_window_z_dur):
    speedup = baseline_avg / durw
    helper.print_markdown(nw, speedup)

