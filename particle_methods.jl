function calc_vortex_solutions(ics, p, t_end; kwargs...)
    (; nv) = p
    ics = vec(stack(collect.(ics)))
    println(ics)
    vortex_sols =
        begin
            sols = VortexCalcs.calc_parallel_ode(
                VortexCalcs.point_vortex_ode!,
                [ics],
                (0, t_end),
                (p..., np=0);
                kwargs...
            ) |> only
            map(i -> sols[2i-1:2i, :], 1:nv)
        end
    return vortex_sols
end

function calc_particle_solutions(vortex_ics, particle_ics, p, t_end; kwargs...)
    (; nv) = p
    vortex_ics = vec(stack(vortex_ics))
    ics = map(ic -> vec(stack([vortex_ics; collect(ic)])), particle_ics)
    particle_sols =
        begin
            sols = VortexCalcs.calc_parallel_ode(
                VortexCalcs.point_vortex_ode!,
                ics,
                (0, t_end),
                p;
                kwargs...
            )
            map(sols) do sol
                sol[2nv+1:end, :]
            end
        end
    return particle_sols
end


function calc_particle_poincare(ics, p, t, numpoints, hyperplane; kwargs...)
    (; nv) = p
    condition1!(u, t, integrator) = begin
        x = @view u[1:2:2nv]
        y = @view u[2:2:2nv]
        return hyperplane(x, y)
    end
    affect1!(integrator) =
        if (length(integrator.sol) % 20 == 0)
            @printf("%2i, %5i\n", Threads.threadid(), length(integrator.sol))
        end
    condition2!(u, t, integrator) = length(integrator.sol) >= numpoints
    affect2!(integrator) = ODE.terminate!(integrator)
    cb1 = ODE.ContinuousCallback(condition1!,
        affect1!,
        nothing,
        save_positions=(true, false)
    )
    cb2 = ODE.DiscreteCallback(condition2!,
        affect2!,
        save_positions=(false, false)
    )
    cbset = ODE.CallbackSet(cb1, cb2)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            reltol=0,
            save_everystep=false,
            save_start=false,
            save_end=false,
            callback=cbset,
            maxiters=1e32,
        )
    )
    poincaremaps = map(
        sol -> sol[2nv+1:end, :],
        calc_parallel_ode(point_vortex_ode!, ics, (0, t), p; kwargs...)
    )
    return poincaremaps
end

function calc_particle_lyapunov_spectrum(ics, p, t, n_steps; kwargs...)
    return calc_parallel_lyapunov_spectrum(point_vortex_ode_tangent!, ics, p, 2, t, n_steps; kwargs...)
end

function calc_particle_ftle(ics, p, t; kwargs...)
    return calc_parallel_ftle(point_vortex_ode_tangent!, ics, p, 2, t; kwargs...)
end


function calc_particle_ld(ics, p, tau; kwargs...)
    (; nv, np) = p
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    ics = map(x -> [x; zeros(np)], ics)
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
    ics = map(x -> [vortex_ics; collect(x); 0], ics)
    sol = calc_parallel_ode(point_vortex_ode_arclength!, ics, (0, t), p; kwargs...)
    arclengths = map(x -> x[end, :], sol)
    return arclengths
end

function calc_particle_average_distance(ics, p, t; kwargs...)
    (; nv, np) = p
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    ics = map(x -> [x; 0; x[2nv+1:end]], ics)
    sol = calc_parallel_ode(point_vortex_ode_particle_avg_distance!, ics, (0, t), p; kwargs...)
    avg_distance = map(x -> x[2(nv+np)+1, :] ./ t, sol)
    return avg_distance
end


function calc_particle_distances_to_vortices(ics, p, t; kwargs...)
    (; nv) = p
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=true,
            save_end=true,
        )
    )
    sols = calc_parallel_ode(point_vortex_ode!, ics, (0, t), p; kwargs...)
    distances = map(sols) do sol
        map(eachcol(sol)) do x
            vortex_x = reshape(x[1:2nv], 2, nv) |> eachcol |> Vector
            particle_x = x[end-1:end]
            map(y -> norm(y - particle_x), vortex_x)
        end
    end
    return distances
end


function calc_particle_velocity_magnitude_field(vortex_points, particle_points, p)
    (; omega, alpha, beta) = p
    vortex_x = getindex.(vortex_points, 1)
    vortex_y = getindex.(vortex_points, 2)
    field = (x) -> norm(particle_velocity(x[1], x[2], vortex_x, vortex_y, omega, alpha, beta))
    return map(field, particle_points)
end

function calc_particle_hamiltonian(vortex_points, particle_points, p)
    (; omega, alpha, beta) = p
    vortex_x = getindex.(vortex_points, 1)
    vortex_y = getindex.(vortex_points, 2)
    field = (x) -> particle_hamiltonian(x[1], x[2], vortex_x, vortex_y, omega, alpha, beta)
    return map(field, particle_points)
end
