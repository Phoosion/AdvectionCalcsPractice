function calc_method_in_parallel(method, ics)
    funcname = join(getproperty.(StackTraces.stacktrace()[1:2], :func), " ")
    tasks = map(eachindex(ics)) do i
        Threads.@spawn begin
            @printf("%s || thread %3i, %8i\n", funcname, Threads.threadid(), i)
            method(ics[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end

function calc_ode(f, ic, tspan, p; kwargs...)
    prob = ODE.ODEProblem(f, ic, tspan, p)
    sol = ODE.solve(prob; kwargs...).u |> stack
    return sol
end

function calc_parallel_ode(f, ics, tspan, p; kwargs...)
    ode_sol = (ic) -> calc_ode(f, ic, tspan, p; kwargs...)
    return calc_method_in_parallel(ode_sol, ics)
end

# function calc_lyapunov_spectrum(f, ic, p, dims, t_end, n; kwargs...)
#     kwargs = merge(
#         NamedTuple(kwargs),
#         (
#             save_everystep=false,
#             save_start=false,
#             save_end=false,
#         )
#     )
#     dt = t_end / n
#     E = Matrix{Float64}(I, dims, dims) |> vec
#     pertub_index = length(ic)+1:(length(ic)+dims^2)
#     prob = ODE.ODEProblem(f, [ic; E], (0, t_end), p)
#     integrator = ODE.init(prob; kwargs...)
#     lambda = zeros(dims)
#     for _ = 1:n
#         ODE.step!(integrator, dt, true)
#         u = integrator.u
#         M = reshape(u[pertub_index], dims, dims)
#         Q, R = qr!(M)
#         lambda += log.(abs.(diag(R)))
#         u[pertub_index] = vec(Matrix(Q))
#         ODE.set_u!(integrator, u)
#     end
#     return lambda ./ t_end
# end

function calc_ftle(f, ic, p, dims, t_end; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=true,
        )
    )
    E = Matrix{Float64}(I, dims, dims) |> vec
    pertub_index = length(ic)+1:(length(ic)+dims^2)
    ic_pertub = [ic; E]
    sol = calc_ode(f, ic_pertub, (0, t_end), p; kwargs...)
    lambdas = map(eachcol(sol)) do x
        M = reshape(x[pertub_index], dims, dims)
        log(opnorm(M)) / t_end
    end
    return lambdas
end

# function calc_parallel_lyapunov_spectrum(f, ics, p, dims, t_end, n; kwargs...)
#     lyap_sol = (ic) -> calc_lyapunov_spectrum(f, ic, p, dims, t_end, n; kwargs...)
#     return calc_parallel_method(lyap_sol, ics)
# end


function calc_parallel_ftle(f, ics, p, dims, t_end; kwargs...)
    ftle_sol = (ic) -> calc_ftle(f, ic, p, dims, t_end; kwargs...)
    return calc_method_in_parallel(ftle_sol, ics)
end
