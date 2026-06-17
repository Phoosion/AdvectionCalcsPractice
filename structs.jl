@kwdef struct PointVortexODEParameters
    vortex_ic::Array{Tuple{Float64,Float64},1}
    omega::Array{Float64,1}
    alpha::Float64 = 0
    beta::Float64 = 0
    tspan::Tuple{Float64,Float64} = (0, 100)
    nsteps::Int = 100
end

@kwdef mutable struct ParticleInitialConditionGrid
    nx::Int = 2^4 + 1
    ny::Int = 2^4 + 1
    intervals::Tuple{Float64,Float64,Float64,Float64} = (-1, 1, -1, 1)
    particle_ic::Array{Tuple{Float64,Float64},1}
    ParticleInitialConditionGrid(nx, ny, intervals) = new(
        nx, ny, intervals,
        Iterators.product(
            collect(range(intervals[1], intervals[2], nx)),
            collect(range(intervals[3], intervals[4], ny))
        ) |> collect |> vec
    )

end

@kwdef struct PointVortexSolution
    initial_parameters::PointVortexODEParameters
    nvortex::Int
    nparticle::Int
    t::Array{Float64,1}
    vortex_solution::Array{Matrix{Float64},1}
    particle_solution::Union{Array{Matrix{Float64},1},Nothing}
    message::String
end

@kwdef struct PoincareMapSolution
    initial_parameters::PointVortexODEParameters
    nvortex::Int
    nparticle::Int
    npoints::Int
    t::Array{Array{Float64,1},1}
    vortex_maps::Array{Matrix{Float64},1}
    particle_maps::Array{Matrix{Float64},1}
    message::String
end

@kwdef struct FTLEField
    initial_parameters::PointVortexODEParameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end

@kwdef struct LDField
    initial_parameters::PointVortexODEParameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end


@kwdef struct DistanceField
    initial_parameters::PointVortexODEParameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end

@kwdef struct DistanceArclengthRatioField
    initial_parameters::PointVortexODEParameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end

@kwdef struct RecurrenceRateField
    initial_parameters::PointVortexODEParameters
    grid::ParticleInitialConditionGrid
    tol::Float64
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end
