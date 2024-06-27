#!/usr/bin/env -S julia -O3

const s	    = 2
const Nmax  = 10000000000

function   𝜁(s,Nstart,Nend)
    _sum    = 0.0
    for n in Nstart:Nend
        _sum += (1.0*n)^-s
    end
    _sum
end

Nthr	    = Threads.nthreads()
print("Number of threads = $(Nthr)\n")
𝜁_vect	    = zeros(Float64,Nthr)
N_per	    = convert(Int64,ceil(Nmax/Nthr))
print("compilation run: $(  𝜁(s,1,100000) )\n")


# run the function in parallel across cores
@time Threads.@threads for thr in 1:Nthr
    Nstart	= 1 + (thr-1) * N_per
    Nstop	= 0 + (thr-0) * N_per
    𝜁_vect[thr] =  𝜁(s,Nstart,Nstop)
end
pso6	    = sum( 𝜁_vect)

_pi = sqrt(6.0 * pso6)

print("π^2 / 6 = $( pso6 )\n")
print("π       = $( _pi  )\n")
