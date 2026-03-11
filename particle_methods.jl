function calc_particle_poincare(ics, p, t, numpoints, hyperplane; kwargs...)
    (; nv) = p
    condition1!(u, t, integrator) = begin
        x = @view u[1:2:end]
        y = @view u[2:2:end]
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

function calc_particle_trajectories(ics, p, tspan; kwargs...)
    (; nv) = p
    sols = calc_parallel_ode(point_vortex_ode!, ics, tspan, p; kwargs...)
    particlesols = begin
        temp = vcat(map(sol -> sol[2*nv+1:end, :], sols)...)
        tempX = @view temp[1:2:end, :]
        tempY = @view temp[2:2:end, :]
        map(rows -> vcat(rows[1]', rows[2]'), zip(eachrow(tempX), eachrow(tempY)))
    end
    return particlesols
end

function calc_particle_lyapunov_spectrum(ics, p, t, n_steps; kwargs...)
    return calc_parallel_lyapunov_spectrum(point_vortex_ode_tangent!, ics, p, 2, t, n_steps; kwargs...)
end

function calc_particle_ftle(ics, p, t; kwargs...)
    return calc_parallel_ftle(point_vortex_ode_tangent!, ics, p, 2, t; kwargs...)
end


function calc_particle_ld(ics, p, tau; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    ics = map(x -> [x; 0], ics)
    ld_forward = map(sol -> sol[end, end], calc_parallel_ode(point_vortex_ode_ld!, ics, (0, tau), p; kwargs...))
    ld_backward = map(sol -> sol[end, end], calc_parallel_ode(point_vortex_ode_ld!, ics, (0, -tau), p; kwargs...))
    return ld_forward + ld_backward
end

function calc_particle_distance(ics, p, t, r; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=true,
            save_end=true,
        )
    )
    sols = calc_parallel_ode(point_vortex_ode!, ics, (0, t), p; kwargs...)
    dist = map(x -> norm(x[end-1:end, end] - x[end-1:end, 1], r), sols)
    return dist
end

function calc_particle_arclength(ics, p, t; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    ics = map(x -> [x; 0], ics)
    sol = calc_parallel_ode(point_vortex_ode_arclength!, ics, (0, t), p; kwargs...)
    arclengths = map(x -> x[end, end], sol)
    return arclengths
end


function calc_particle_basins_of_attraction(ics, p, t; kwargs...)
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
    attractions = map(sols) do sol
        distances = map(eachcol(sol)) do x
            x_vortex = reshape(x[1:2nv], 2, nv) |> eachcol |> Vector
            x_particle = x[end-1:end]
            map(y -> norm(y - x_particle), x_vortex)
        end
        map(i -> distances[i] ./ distances[1], eachindex(distances))
    end
    return attractions
end


function calc_multiple_particle_methods(vortex_ics, particle_ics, p, t, flags, saves; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=true,
            saveat=saves,
            save_end=true,
        )
    )
    (;
        vortex_sol_flag,
        ftle_flag, ld_flag,
        arclength_flag, distance_flag,
        attraction_flag
    ) = flags

    (; omega, alpha, beta, gamma) = p

    ics = vec.(stack.(map(
        ic -> [
            vortex_ics;
            ic
        ], particle_ics
    )))

    ics_extended = map(
        ic -> [ic;
            (ftle_flag) ? Matrix{Float64}(I, 2, 2) |> vec : [];
            (arclength_flag) ? [0.0] : []
        ], ics
    )

    (; nv, np) = p
    n = nv + np
    n_tangents = ftle_flag ? 2np : 0
    n_arclengths = arclength_flag ? np : 0

    p_extended = (
        p...,
        nv=nv, np=np,
        n_tangents=n_tangents,
        n_arclengths=n_arclengths,
    )

    vortex_ics = vortex_ics |> stack |> vec

    vortex_sol = vortex_sol_flag ? VortexCalcs.calc_ode(
        VortexCalcs.point_vortex_ode!,
        vortex_ics,
        (0, t), (omega, alpha, beta);
        kwargs...
    ) : []

    sols = calc_parallel_ode(point_vortex_ode_combined!, ics_extended, (0, t), p_extended; kwargs...)

    particle_sols = map(sol -> sol[2nv+1:2nv+2, :], sols)

    ftle_sols = ftle_flag ? map(sols) do sol
        map(tangents -> log(opnorm(reshape(tangents, 2, 2))) / t, eachcol(sol[2n+1:2(n+n_tangents), :]))
    end : []

    arclength_sols = arclength_flag ? map(sol -> vec(sol[2(n+n_tangents)+1:2(n+n_tangents)+n_arclengths, :]), sols) : []

    distance_sols = distance_flag ? map(particle_sols) do particle_sol
        sol_cols = eachcol(particle_sol)
        map(i -> norm(sol_cols[i] - sol_cols[1]), eachindex(sol_cols))
    end : []

    attraction_sols = attraction_flag ? map(sols) do sol
        distances = map(eachcol(sol)) do x
            particle_x = x[2nv+1:2(nv+np)]
            vortex_x = reshape(x[1:2nv], 2, nv) |> eachcol |> Vector
            map(y -> norm(y - particle_x), vortex_x)
        end
        map(i -> distances[i] ./ distances[1], eachindex(distances))
    end : []

    ld_sols = ld_flag ? begin
        ics_ld = map(ic -> [ic; 0], ics)
        ld_forward = map(sol -> sol[end, :], calc_parallel_ode(point_vortex_ode_ld!, ics_ld, (0, t), p; kwargs..., saveat=saves))
        ld_backward = map(sol -> sol[end, :], calc_parallel_ode(point_vortex_ode_ld!, ics_ld, (0, -t), p; kwargs..., saveat=-saves))
        ld_forward + ld_backward
    end : []

    return (
        vortex_sol=vortex_sol,
        particle_sol=particle_sols,
        ftle_sol=ftle_sols,
        ld_sol=ld_sols,
        arclength_sol=arclength_sols,
        distance_sol=distance_sols,
        attraction_sol=attraction_sols
    )
end