#!/usr/bin/env -S julia

using SpecialFunctions, CairoMakie

const N=1000

const xmin      = -0.50
const xmax      =  2.0
const ymin      = -0.5
const ymax      =  2.0

dx = (xmax-xmin)/N
dy = (ymax-ymin)/N


function fill_z!(z::Array{Complex{Float64}},N::Int64,M::Int64,xmin::Float64,dx::Float64,ymin::Float64,dy::Float64)
        [@inbounds z[j,i]=ComplexF64(xmin+(i-1)*dx,ymin+(j-1)*dy) for j in 1:N, i in 1:N]
        return nothing
end

function main() 

x,y = collect(xmin:dx:xmax-dx),collect(ymin:dy:ymaxi-dy)
z = zeros(ComplexF64,N,N)
fill_z!(z,N,N,xmin,dx,ymin,dy)
ζ = zeta.(z)
mζ = abs.(ζ)
aζ = angle.(ζ)

surface(mζ, colormap = :deep)

end
