include("methods.jl")
include("parallel_methods.jl")

function calc_vortex_solutions(ic, p, t_end; kwargs...)
    (; nv) = p
    ###
    @assert(length(ic) == nv)
    ###
    ic = vec(stack(collect.(ic)))
    vortex_sols = begin
        sols = calc_ode(
            point_vortex_ode!,
            ic,
            (0, t_end),
            (p..., np=0);
            kwargs...
        ) |> only
        map(i -> sols[2i-1:2i, :], 1:nv)
    end
    return vortex_sols
end

function calc_vortex_recurrence_plot(
    ic, p, t;
    tol=1e-6,
    coords::Union{Nothing,Array{Int,1}}=nothing,
    kwargs...
)
    funcname = join(getproperty.(StackTraces.stacktrace()[1:2], :func), " ")
    (; nv, np) = p
    ###
    (np == 0) || throw(ArgumentError("np != 0"))
    (length(ic) == nv) || throw(ArgumentError("length(ic) != nv"))
    (isnothing(coords) || coords == [1] || coords == [2]) || throw(ArgumentError("Invalid coords"))
    ###
    ic = ic |> Iterators.flatten |> collect
    coords = isnothing(coords) ? [1, 2] : coords
    return map(fetch,
        map(1:nv) do i
            Threads.@spawn begin
                Printf.@printf("%s || thread %3i, %8i\n", funcname, Threads.threadid(), i)
                calc_recurrence_plot(point_vortex_ode!, ic, p, t, tol; axis=[[2i - 1; 2i][coords]], kwargs...)
            end
        end
    )
end

function calc_particle_recurrence_plot(
    vortex_ics, particle_ics, p, t;
    tol=1e-6,
    coords::Union{Nothing,Array{Int,1}}=nothing,
    kwargs...
)
    funcname = join(getproperty.(StackTraces.stacktrace()[1:2], :func), " ")
    (; nv, np) = p
    ###
    (np == 1) || throw(ArgumentError("np != 1"))
    (isnothing(coords) || coords == [1] || coords == [2]) || throw(ArgumentError("Invalid coords"))
    (length(vortex_ics) == nv) || throw(ArgumentError("length(ic) != 2nv"))
    ###
    vortex_ics = vec(stack(vortex_ics))
    ics = map(ic -> vec(stack([vortex_ics; ic])), collect.(particle_ics))
    coords = isnothing(coords) ? [1, 2] : coords
    axis = [2nv + 1; 2nv + 2][coords]
    return map(fetch,
        map(enumerate(ics)) do (i, ic)
            Threads.@spawn begin
                Printf.@printf("%s || thread %3i, %8i\n", funcname, Threads.threadid(), i)
                calc_recurrence_plot(point_vortex_ode!, ic, p, t, tol; axis=[axis], kwargs...)
            end
        end
    )
end

function calc_particle_recurrence_rate(
    vortex_ics, particle_ics, p, t;
    tol=1e-3 / 2,
    coords::Union{Nothing,Array{Int,1}}=nothing,
    kwargs...
)
    funcname = join(getproperty.(StackTraces.stacktrace()[1:2], :func), " ")
    ###
    (isnothing(coords) || coords == [1] || coords == [2]) || throw(ArgumentError("Invalid coords"))
    ###
    sols = calc_particle_solutions(vortex_ics, particle_ics, p, t[end]; (kwargs..., saveat=t)...)
    coords = isnothing(coords) ? [1, 2] : coords
    return map(fetch,
        map(enumerate(sols)) do (i, sol)
            Threads.@spawn begin
                Printf.@printf("%s || thread %3i, %8i\n", funcname, Threads.threadid(), i)
                calc_recurrence_rate_from_data((@view sol[coords, :]), tol)
            end
        end
    )
end


function calc_vortex_poincare(ic, p, hyperplane, n_points; t_max=1e6, kwargs...)
    (; nv) = p
    ###
    @assert(length(ic) == nv)
    ###
    ic = vec(stack(collect.(ic)))
    t, vortex_map = calc_poincare_map(point_vortex_ode!, ic, (p..., np=0), hyperplane, n_points; t_max=t_max)
    return (Float64.(t), Float64.(vortex_map))
end

function calc_vortex_particle_poincare(vortex_ics, particle_ics, p, hyperplane, n_points; t_max=1e6, kwargs...)
    (; nv, np) = p
    ###
    @assert(length(vortex_ics) == nv & np == 1)
    ###
    vortex_ics = vec(stack(vortex_ics))
    ics = map(ic -> vec(stack([vortex_ics; ic])), collect.(particle_ics))
    poincare_maps = begin
        sols = calc_parallel_poincare_map(point_vortex_ode!, ics, p, hyperplane, n_points; t_max=t_max)
        map(sols) do (t, map)
            return (t, map[1:2nv, :], map[2nv+1:end, :])
        end
    end
    return poincare_maps
end

function calc_particle_solutions(vortex_ics, particle_ics, p, t_end; kwargs...)
    funcname = join(getproperty.(StackTraces.stacktrace()[1:2], :func), " ")
    (; nv, np) = p
    ###
    (np == 1) || throw(ArgumentError("np != 1"))
    (length(vortex_ics) == nv) || throw(ArgumentError("length(ic) != 2nv"))
    ###
    vortex_ics = vec(stack(vortex_ics))
    ics = map(ic -> vec(stack([vortex_ics; ic])), collect.(particle_ics))
    particle_sols = begin
        sols = VortexCalcs.calc_parallel_ode(
            point_vortex_ode!,
            ics,
            (0, t_end),
            p;
            kwargs...
        )
        @views map(sols) do sol
            sol[2nv+1:end, :]
        end
    end
    return particle_sols
end

function calc_particle_ftle(vortex_ics, particle_ics, p, t; kwargs...)
    vortex_ics = vec(stack(vortex_ics))
    ics = map(ic -> vec(stack([vortex_ics; ic])), collect.(particle_ics))
    return calc_parallel_ftle(point_vortex_ode_tangent!, ics, p, 2, t; kwargs...)
end

function calc_particle_finite_diff_ftle(vortex_ics, particle_ics, p, t; kwargs...)
    (; nv, np) = p
    ###
    @assert(np == 1)
    ###
    vortex_ics = vec(stack(vortex_ics))
    ics = map(ic -> vec(stack([vortex_ics; ic])), collect.(particle_ics))
    sol_axis = [1, 2] .+ 2 * nv
    return calc_parallel_finite_diff_ftle(VortexCalcs.point_vortex_ode!, ics, p, t; sol_axis=sol_axis, kwargs...)
end

function calc_particle_ld(vortex_ics, particle_ics, p, tau; kwargs...)
    (; nv, np) = p
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    vortex_ics = vec(stack(vortex_ics))
    ics = map(ic -> vec(stack([vortex_ics; ic; zeros(np)])), collect.(particle_ics))
    ld_forward = map(calc_parallel_ode(point_vortex_ode_ld!, ics, (0, tau), p; kwargs...)) do sol
        sol[2(nv+np)+1:2(nv+np)+np, :]
    end
    ld_backward = map(calc_parallel_ode(point_vortex_ode_ld!, ics, (0, -tau), p; kwargs...)) do sol
        sol[2(nv+np)+1:2(nv+np)+np, :]
    end
    return ld_forward .+ ld_backward
end

function calc_particle_distance(vortex_ics, particle_ics, p, t; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=true,
            save_end=true,
        )
    )
    sols = calc_particle_solutions(vortex_ics, particle_ics, p, t; kwargs...)
    dist = map(sols) do sol
        x = eachcol(sol)
        map(i -> norm(x[i] - x[1]), eachindex(x))
    end
    return dist
end

function calc_particle_arclength(vortex_ics, particle_ics, p, t; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    vortex_ics = stack(collect.(vortex_ics)) |> vec
    ics = map(x -> [vortex_ics; x; 0; x], collect.(particle_ics))
    sol = calc_parallel_ode(point_vortex_ode_arclength!, ics, (0, t), p; kwargs...)
    arclengths = map(x -> x[end, :], sol)
    return arclengths
end

function calc_particle_average_distance(vortex_ics, particle_ics, p, t; kwargs...)
    (; nv, np) = p
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    vortex_ics = stack(collect.(vortex_ics)) |> vec
    ics = map(x -> [vortex_ics; collect(x); 0], ics)
    ics = map(x -> [x; 0; x[2nv+1:end]], ics)
    sol = calc_parallel_ode(point_vortex_ode_particle_avg_distance!, ics, (0, t), p; kwargs...)
    avg_distance = map(x -> x[2(nv+np)+1, :] ./ t, sol)
    return avg_distance
end


# function calc_particle_distances_to_vortices(ics, p, t; kwargs...)
#     (; nv) = p
#     kwargs = merge(
#         NamedTuple(kwargs),
#         (
#             save_everystep=false,
#             save_start=true,
#             save_end=true,
#         )
#     )
#     sols = calc_parallel_ode(point_vortex_ode!, ics, (0, t), p; kwargs...)
#     distances = map(sols) do sol
#         map(eachcol(sol)) do x
#             vortex_x = reshape(x[1:2nv], 2, nv) |> eachcol |> Vector
#             particle_x = x[end-1:end]
#             map(y -> norm(y - particle_x), vortex_x)
#         end
#     end
#     return distances
# end


function calc_particle_velocity_magnitude_field(vortex_points, particle_points, p)
    (; omega, alpha, beta) = p
    vortex_x = getindex.(vortex_points, 1)
    vortex_y = getindex.(vortex_points, 2)
    field = (x) -> norm(particle_velocity(
        x[1], x[2],
        vortex_x, vortex_y,
        omega, alpha, beta
    ))
    return map(field, particle_points)
end

function calc_particle_hamiltonian(vortex_points, particle_points, p)
    (; omega, alpha, beta) = p
    vortex_x = getindex.(vortex_points, 1)
    vortex_y = getindex.(vortex_points, 2)
    field = (x) -> particle_hamiltonian(
        x[1], x[2],
        vortex_x, vortex_y,
        omega, alpha, beta
    )
    return map(field, particle_points)
end
