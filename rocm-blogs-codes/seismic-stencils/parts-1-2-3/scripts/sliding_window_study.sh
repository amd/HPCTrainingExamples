# Load helper functions
source scripts/helper.sh
###################################################################################################
# Initial workload settings for this study
study=sliding_window_2023_04_28
name=vary_nw
nt=11
align=64
use_offset=1
radius=4 
vec=0

###################################################################################################
begin "Parameter study for nw (small grid)"
nx=400
ny=400
nz=400

# This grid size is currently not used in the blog post.
#vary_nw 1 1 1
#vary_nw 0 100 2
#vary_nw 100 410 10
end
###################################################################################################
begin "Parameter study for nw (large grid)"
nx=1000
ny=1000
nz=1000
vary_nw 1 10 1
vary_nw 10 20 2
vary_nw 20 40 4
vary_nw 40 80 8
vary_nw 80 200 16
vary_nw 200 1032 32
vary_nw 968 1032 32
vary_nw 500 501 1
end
###################################################################################################
