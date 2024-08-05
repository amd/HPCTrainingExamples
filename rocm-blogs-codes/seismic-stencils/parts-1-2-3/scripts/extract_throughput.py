import helper

baseline_kernel = "compute_fd_z_gpu_kernel"
kernel = "compute_fd_sliding_window_z_gpu_kernel"

r=4

fn = lambda align, use_offset, n: helper.filename("logs/baseline_2023_04_28", n, n, n,
        "throughput_curve",
        r, align, n, use_offset)

files = []
ns = []
for n in range(64, 1200, 10):
    files.append(fn(1, 0, n))
    ns.append(n)

afiles = []
for n in range(64, 1200, 10):
    afiles.append(fn(64, 1, n))


fd_z = helper.read_kernel_seq(files, baseline_kernel, "Duration")
window_fd_z = helper.read_kernel_seq(files, kernel, "Duration")
afd_z = helper.read_kernel_seq(afiles, baseline_kernel, "Duration")
awindow_fd_z = helper.read_kernel_seq(afiles, kernel, "Duration")

helper.print_csv("Grid size", "fd_z ", "fd_z (aligned)", "window fd_z" , "window fd_z (aligned)")
for n, t, at, t_window, at_window in zip(ns, fd_z, afd_z, window_fd_z, awindow_fd_z):
    helper.print_csv(n, 
            helper.gcells(t, n**3), 
            helper.gcells(at, n**3), 
            helper.gcells(t_window, n**3),
            helper.gcells(at_window, n**3)
            )
