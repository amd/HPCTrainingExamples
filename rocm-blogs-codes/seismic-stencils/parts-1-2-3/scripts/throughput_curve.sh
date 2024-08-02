# Load helper functions
source scripts/helper.sh
###################################################################################################
# Workload 
nx=1000
ny=1000
nz=1000
nt=11
radius=4
vec=0
name=throughput_curve
study=baseline_2023_04_28

###################################################################################################
begin "Throughput curve with alignment optimization"
align=64
use_offset=1
vary_grid_size 64 1200 10
end

begin "Throughput curve without alignment optimization"
align=1
use_offset=0
vary_grid_size 64 1200 10
end
###################################################################################################
