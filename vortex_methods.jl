function calc_particle_poincare(ICs, p, T, numpoints, hyperplane; kwargs...)
    nv = length(omega)
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
    poincaremap = calc_parallel_ode(point_vortex_ode!, ICs, (0, T), p; kwargs...)
    map!(res -> res[2nv+1:end, :], poincaremap)
    return poincaremap
end

function calc_particle_trajectories(ICs, p, tspan; kwargs...)
    nv = length(p[1])
    sol = calc_parallel_ode(point_vortex_ode!, ICs, tspan, p; kwargs...)
    particlesol = begin
        temp = vcat(map(res -> res[2*nv+1:end, :], sol)...)
        tempX = @view temp[1:2:end, :]
        tempY = @view temp[2:2:end, :]
        map(rows -> vcat(rows[1]', rows[2]'), zip(eachrow(tempX), eachrow(tempY)))
    end
    return particlesol
end

function calc_particle_lyapunov_spectrum(ICs, p, T, N; kwargs...)
    return calc_lyapunov_spectrum(point_vortex_ode_tangent!, ICs, p, 2, T, N; kwargs...)
end

function calc_particle_ftle(ICs, p, T; kwargs...)
    return calc_parallel_ftle(point_vortex_ode_tangent!, ICs, p, 2, T; kwargs...)
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
    ICs = map(x -> [x; 0], ics)
    LD_forward = map(sol -> sol[end, end], calc_parallel_ode(point_vortex_ode_ld!, ICs, (0, tau), p; kwargs...))
    LD_backward = map(sol -> sol[end, end], calc_parallel_ode(point_vortex_ode_ld!, ICs, (0, -tau), p; kwargs...))
    return LD_forward + LD_backward
end

function calc_particle_distance(ICs, p, T, r; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=true,
            save_end=true,
        )
    )
    sol = calc_parallel_ode(point_vortex_ode!, ICs, (0, T), p; kwargs...)
    dist = map(x -> norm(x[end-1:end, end] - x[end-1:end, 1], r), sol)
    return dist
end

function calc_particle_arclength(ICs, p, T; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    ICs = map(x -> [x; 0], ICs)
    sol = calc_parallel_ode(point_vortex_ode_arclength!, ICs, (0, T), p; kwargs...)
    dist = map(x -> x[end, end], sol)
    return dist
end