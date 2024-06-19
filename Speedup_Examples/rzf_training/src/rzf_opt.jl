#!/usr/bin/env -S julia -O3
using LoopVectorization
const s	    = 2
const Nmax  = 10000000000

function   𝜁(s) 
    _sum    = 0.0
    @turbo for n in 1:Nmax
	    _sum += (1.0*n)^-s
    end
    _sum
end

@time pso6  =  𝜁(2)
_pi         = sqrt(6.0 * pso6)
print("π^2 / 6 = $( pso6 )\n")
print("π       = $( _pi  )\n")
