import tomli
import helper

nx=1000
ny=1000
nz=1000
r=4
# With proper padding
align=64
use_offset=1
peak=helper.peak
size = nx * ny * nz
ideal = 4 * size / 1e6

fn = lambda align, use_offset: helper.filename("logs/baseline_2023_04_26", nx, ny, nz, "baseline",
        r, align, 1, use_offset)


# Unaligned data
file=fn(1, 0)
fd_z = helper.read_kernel_data(file, "compute_fd_z_gpu_kernel")
dur = fd_z["Duration"].mean
l1 = fd_z["L1"].mean
l2r = fd_z["L2 Read"].mean
l2w = fd_z["L2 Write"].mean
hbmr = fd_z["HBM Read"].mean
hbmw = fd_z["HBM Write"].mean
bw = fd_z["HBM Bandwidth"].mean
# Aligned data
afile=fn(align, use_offset)
afd_z = helper.read_kernel_data(afile, "compute_fd_z_gpu_kernel")
adur = afd_z["Duration"].mean
al1 = afd_z["L1"].mean
al2r = afd_z["L2 Read"].mean
al2w = afd_z["L2 Write"].mean
ahbmr = afd_z["HBM Read"].mean
ahbmw = afd_z["HBM Write"].mean
abw = afd_z["HBM Bandwidth"].mean

kernel_label = f"Baseline R={r} <64x8> Unaligned"
akernel_label = f"Baseline R={r} <64x8> Aligned" 

helper.print_markdown_header("Kernel", "Throughput (Gcells/s)", "Bandwidth (PoP)",  "Speedup")
helper.print_markdown(kernel_label, helper.gcells(dur, size), helper.bwpop(bw), 1.0)
helper.print_markdown(akernel_label, helper.gcells(adur, size), helper.bwpop(abw), dur / adur)
print("")
helper.print_markdown_header("Kernel", "L1", "L2",  "HBM", "")
helper.print_markdown(kernel_label, l1 / ideal, (l2r + l2w) / ideal, (hbmr + hbmw) / ideal )
helper.print_markdown(akernel_label, al1 / ideal, (al2r + al2w) / ideal, (ahbmr + ahbmw) / ideal )
print("")

helper.print_markdown_header("Kernel", "HBM Read / Write")
helper.print_markdown(kernel_label, hbmr / hbmw )
helper.print_markdown(akernel_label, ahbmr / ahbmw )
print("")
