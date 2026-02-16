module VortexCalcs

using LinearAlgebra
using Printf
import OrdinaryDiffEq as ODE

export
    calc_particle_poincare,
    calc_particle_trajectories,
    calc_particle_ftle,
    calc_particle_ld,
    calc_particle_distance,
    calc_ode,
    point_vortex_ode!

include("./ode_calc_methods.jl")
include("./vortex_rhs.jl")
include("./vortex_methods.jl")

end