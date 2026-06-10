@kwdef struct PointVortexODEParameters
    vortex_ic::Array{Tuple{Float64,Float64},1}
    omega::Array{Float64,1}
    alpha::Float64 = 0
    beta::Float64 = 0
    tspan::Tuple{Float64,Float64} = (0, 100)
    nsteps::Int = 100
end

@kwdef struct ParticleInitialConditionGrid
    grid_nx::Int = 2^4 + 1
    grid_ny::Int = 2^4 + 1
    xy_interval::Tuple{Float64,Float64,Float64,Float64} = (-1, 1, -1, 1)
end

@kwdef struct PointVortexSolution
    initial_parameters::Parameters
    nvortex::Int
    nparticle::Int
    solution::Array{Matrix{Float64},1}
    message::String
end

@kwdef struct PoincareMapSolution
    initial_parameters::Parameters
    times::Array{Float64,1}
    solution::Array{Matrix{Float64},1}
    message::String
end

@kwdef struct FTLEField
    initial_parameters::Parameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end

@kwdef struct LDField
    initial_parameters::Parameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end


@kwdef struct DistanceField
    initial_parameters::Parameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end

@kwdef struct DistanceArclengthRatioField
    initial_parameters::Parameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end

@kwdef struct RecurrenceRateField
    initial_parameters::Parameters
    grid::ParticleInitialConditionGrid
    solution::Union{Matrix{Float64},Array{Float64,1}}
    message::String
end
