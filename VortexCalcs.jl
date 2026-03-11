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
    calc_particle_arclength,
    calc_particle_basins_of_attraction,
    calc_particle_lyapunov_spectrum,
    calc_multiple_particle_methods,
    calc_ode,
    point_vortex_ode!

include("./ode_calc_methods.jl")
include("./ode_rhs.jl")
include("./particle_methods.jl")

end