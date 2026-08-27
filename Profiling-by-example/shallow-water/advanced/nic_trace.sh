NIC_STEPS=$(printf "step_%s," {10..29})
NIC_STEPS=${NIC_STEPS%,}

run_nic_trace() {
    local ranks=$1 out=$2 binary=$3
    mpirun -n "$ranks" ${MPI_BIND} ${GPU_BIND} \
        rocprof-sys-run --preset=trace-hpc --flat-profile \
        --selected-regions "$NIC_STEPS" -o "$out" -- "$binary"
}
