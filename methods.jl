function calc_ode(f, ic, tspan, p; kwargs...)
    prob = DE.ODEProblem(f, ic, tspan, p)
    sol = DE.solve(prob; kwargs...).u |> stack
    return sol
end

function calc_ftle(f, ic, p, dims, t_end; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    E = Matrix{Float64}(LinearAlgebra.I, dims, dims) |> vec
    pertub_index = length(ic)+1:(length(ic)+dims^2)
    ic_pertub = [ic; E]
    sol = calc_ode(f, ic_pertub, (0, t_end), p; kwargs...)
    lambda = map(eachcol(sol)) do x
        M = reshape(x[pertub_index], dims, dims)
        log(LinearAlgebra.opnorm(M)) / t_end
    end
    return lambda
end

function calc_finite_diff_ftle(f, ic, p, t_end; sol_axis=nothing, kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    sol_axis = isnothing(sol_axis) ? eachindex(ic) : sol_axis
    sol = (x) -> calc_ode(
        f, setindex!(ic, x, sol_axis), (0, t_end), p; kwargs...
    )[sol_axis]
    delta = FiniteDifferences.jacobian(
        central_fdm(2, 1, max_range=1e-2, factor=1e6),
        sol, ic[sol_axis]
    )[1]
    lambda = log(LinearAlgebra.opnorm(delta)) / t_end
    return lambda
end


function calc_poincare_map(f, ic, p, hyperplane, n_points; t_max=1e6, kwargs...)
    condition1!(u, t, integrator) = begin
        return hyperplane(u)
    end
    affect1!(integrator) = begin
        if (length(integrator.sol.u) % 20 == 0)
            Printf.@printf(
                "%8sthread %2i || n_points = %5i, %.1f\n",
                "", Threads.threadid(), length(integrator.sol.u), integrator.sol.t[end])
        end
    end
    condition2!(u, t, integrator) = length(integrator.sol.u) >= n_points
    affect2!(integrator) = begin
        DE.terminate!(integrator)
        Printf.@printf(
            "%8sthread %2i || %.1f\n", "",
            Threads.threadid(), integrator.sol.t[end]
        )
    end
    cb1 = DE.ContinuousCallback(condition1!,
        affect1!,
        nothing,
        save_positions=(true, false)
    )
    cb2 = DE.DiscreteCallback(condition2!,
        affect2!,
        save_positions=(false, false)
    )
    cbset = DE.CallbackSet(cb1, cb2)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            reltol=0,
            # abstol=1e-12,
            save_everystep=false,
            save_start=false,
            save_end=false,
            callback=cbset,
            maxiters=1e32,
        )
    )
    prob = DE.ODEProblem(f, ic, (0, t_max), p)
    sol = DE.solve(prob; kwargs...)
    return (sol.t, stack(sol.u))
end


function calc_recurrence_plot(
    f, ic, p, t, tol;
    axis::Union{Nothing,Array{Int,1}}=nothing, kwargs...
)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=false,
            saveat=t
        )
    )
    axis = isnothing(axis) ? eachindex(ic) : axis
    u = eachcol(calc_ode(f, ic, (0, t[end]), p; kwargs...)[axis, :])
    rp = [LinearAlgebra.norm(x - y) < tol for x in u, y in u]
    return rp
end

function calc_recurrence_rate(
    f, ic, p, t, tol;
    axis::Union{Nothing,Array{Int,1}}=nothing, kwargs...
)
    n = length(t)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=false,
            saveat=t
        )
    )
    axis = isnothing(axis) ? eachindex(ic) : axis
    u = eachcol(calc_ode(f, ic, (0, t[end]), p; kwargs...)[axis, :])
    rr = sum(x -> sum(y -> LinearAlgebra.norm(x - y) < tol, u), u) / (n^2)
    return rr
end

function calc_recurrence_plot_from_data(u, tol)
    _u = eachcol(u)
    recurrence_plot = [LinearAlgebra.norm(x - y) < tol for x in _u, y in _u]
    return recurrence_plot
end

function calc_recurrence_rate_from_data(u, tol)
    _u = eachcol(u)
    n = length(_u)
    recurrence_rate = sum(x -> sum(y -> LinearAlgebra.norm(x - y) < tol, u), u) / (n^2)
    return recurrence_rate
end
