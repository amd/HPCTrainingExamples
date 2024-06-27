#!/usr/bin/env -S julia -O3

const s	    = 2
const Nmax  = 10000000000

function   𝜁(s) 
    _sum    = 0.0
    for n in 1:Nmax
	_sum += (1.0*n)^-s
    end
    _sum
end

@time _pi = sqrt(6.0 * 𝜁(s))

print("π^2 / 6 = $( 𝜁(2) )")
print("π       = $( _pi  )")
