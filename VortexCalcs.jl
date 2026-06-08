module VortexCalcs

import LinearAlgebra
import Printf
import FiniteDifferences
import DifferentialEquations as DE

export calc_vortex_solutions,
    calc_particle_solutions,
    calc_vortex_poincare,
    calc_particle_ftle,
    calc_particle_finite_diff_ftle,
    calc_particle_ld,
    calc_particle_distance,
    calc_particle_arclength,
    calc_particle_basins_of_attraction,
    calc_ode,
    calc_recurrence_plot,
    particle_velocity,
    point_vortex_ode!


include("./rhs.jl")
include("./methods.jl")
include("./parallel_methods.jl")
include("./particle_methods.jl")
include("./scalar_fields.jl")
end
