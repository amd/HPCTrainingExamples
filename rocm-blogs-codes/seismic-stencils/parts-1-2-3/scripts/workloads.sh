# Load helper functions
source helper.sh
###################################################################################################
# Workload 
nx=1000
ny=1000
nz=1000
nt=11
vec=0
study=baseline_2023_04_28

###################################################################################################
begin "Baseline cases without proper padding and offset"
nw=10
align=1
radius=4
use_offset=0
name=baseline
nw=1 run
nw=10 run
nw=100 run
nw=1000 run
end
###################################################################################################
begin "Baseline cases with proper padding and offset"
align=64
radius=4
use_offset=1
name=baseline
nw=1 run
nw=10 run
nw=100 run
nw=1000 run
end

###################################################################################################
begin "Parameter study for nw"
align=64
use_offset=1

radius=0
radius=0 vary_nw 0 1000 10
radius=4 vary_nw 0 1000 10
end
###################################################################################################
