# Shortened Oceananigans AMDGPU test.
#
# This reproduces the "AMDGPU on LatitudeLongitudeGrid with HydrostaticFreeSurfaceModel"
# testset from Oceananigans' test/test_amdgpu.jl, but is self-contained so it does not
# pull in test/dependencies_for_runtests.jl (which requires CUDA, MPI, and the full test
# harness). Running only this testset keeps the CI check fast.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: architecture
using Oceananigans.Simulations: Simulation, run!, iteration, time
using Test
using AMDGPU
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

float_types = (Float32, Float64)

function build_and_timestep_simulation(model)
    FT = eltype(model)

    for field in merge(model.velocities, model.tracers)
        @test parent(field) isa ROCArray
    end

    simulation = Simulation(model, Δt=1minute, stop_iteration=3, verbose=false)
    run!(simulation)

    @test iteration(simulation) == 3
    @test time(simulation) ≈ FT(3minutes)

    return nothing
end

@testset "AMDGPU on LatitudeLongitudeGrid with HydrostaticFreeSurfaceModel" begin
    roc = AMDGPU.ROCBackend()
    arch = GPU(roc)

    for FT in float_types
        @info " Testing on $arch with $FT"

        grid = LatitudeLongitudeGrid(arch, FT, size=(4, 8, 16), longitude=(-60, 60), latitude=(0, 60), z=(0, 1))

        @test parent(grid.Δxᶜᶜᵃ) isa ROCArray
        @test parent(grid.Δxᶠᶜᵃ) isa ROCArray
        @test parent(grid.Δxᶜᶠᵃ) isa ROCArray
        @test parent(grid.Δxᶠᶠᵃ) isa ROCArray
        @test parent(grid.Azᶜᶜᵃ) isa ROCArray
        @test parent(grid.Azᶠᶜᵃ) isa ROCArray
        @test parent(grid.Azᶜᶠᵃ) isa ROCArray
        @test parent(grid.Azᶠᶠᵃ) isa ROCArray
        @test eltype(grid) == FT
        @test architecture(grid) isa GPU

        equation_of_state = TEOS10EquationOfState()
        buoyancy = SeawaterBuoyancy(; equation_of_state)

        model = HydrostaticFreeSurfaceModel(grid; buoyancy,
            coriolis = FPlane(latitude=45),
            tracers = (:T, :S),
            momentum_advection = WENO(order=5),
            tracer_advection = WENO(order=5),
            free_surface = SplitExplicitFreeSurface(grid; substeps=60))

        build_and_timestep_simulation(model)
    end
end
