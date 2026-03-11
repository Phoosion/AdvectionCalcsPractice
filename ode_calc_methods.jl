function calc_ode(f, ic, tspan, p; kwargs...)
    prob = ODE.ODEProblem(f, ic, tspan, p)
    sol = ODE.solve(prob; kwargs...).u |> stack
    return sol
end

function calc_parallel_ode(f, ics, tspan, p; kwargs...)
    funcname = join(StackTraces.stacktrace()[1:2].func, " ")
    ode_sol = (ic) -> calc_ode(f, ic, tspan, p; kwargs...)

    # chunks = Iterators.partition(ics, cld(length(ics), Threads.nthreads()))
    # tasks = map(chunks) do chunk
    #     Threads.@spawn map(eachindex(chunk)) do i
    #         @printf("%s, thread %3i, %8i\n", funcname, Threads.threadid(), i)
    #         ode_sol(chunk[i])
    #     end
    # end
    # sol = mapreduce(fetch, vcat, tasks)

    tasks = map(eachindex(ics)) do i
        Threads.@spawn begin
            @printf("%s || thread %3i, %8i\n", funcname, Threads.threadid(), i)
            ode_sol(ics[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end

function calc_lyapunov_spectrum(f, ic, p, dims, t_end, n; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=false,
        )
    )
    dt = t_end / n
    E = Matrix{Float64}(I, dims, dims) |> vec
    pertub_index = length(ic)+1:(length(ic)+dims^2)
    prob = ODE.ODEProblem(f, [ic; E], (0, t_end), p)
    integrator = ODE.init(prob; kwargs...)
    lambda = zeros(dims)
    for i = 1:n
        ODE.step!(integrator, dt, true)
        u = integrator.u
        M = reshape(u[pertub_index], dims, dims)
        Q, R = qr!(M)
        lambda += log.(abs.(diag(R)))
        u[pertub_index] = vec(Matrix(Q))
        ODE.set_u!(integrator, u)
    end
    return (ic[end-dims+1:end], lambda ./ t_end)
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
    E = Matrix{Float64}(I, dims, dims) |> vec
    pertub_index = length(ic)+1:(length(ic)+dims^2)
    ic_pertub = [ic; E]
    sol = calc_ode(f, ic_pertub, (0, t_end), p; kwargs...)
    M = reshape(sol[pertub_index, :], dims, dims)
    lambda = log(opnorm(M)) / t_end
    return (ic[end-dims+1:end], lambda)
end

function calc_parallel_lyapunov_spectrum(f, ics, p, dims, t_end, n; kwargs...)
    funcname = join(StackTraces.stacktrace()[1:2].func, " ")
    lyap_sol = (ic) -> calc_lyapunov_spectrum(f, ic, p, dims, t_end, n; kwargs...)

    # chunks = Iterators.partition(ics, cld(length(ics), Threads.nthreads()))
    # tasks = map(chunks) do chunk
    #     Threads.@spawn map(eachindex(chunk)) do i
    #         @printf("%s, thread %2i, %5i\n", funcname, Threads.threadid(), i)
    #         ftle_sol(chunk[i])
    #     end
    # end
    # sol = mapreduce(fetch, vcat, tasks)

    tasks = map(eachindex(ics)) do i
        Threads.@spawn begin
            @printf("%s, thread %3i, %8i\n", funcname, Threads.threadid(), i)
            lyap_sol(ics[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end


function calc_parallel_ftle(f, ics, p, dims, t_end; kwargs...)
    funcname = join(StackTraces.stacktrace()[1:2].func, " ")
    ftle_sol = (ic) -> calc_ftle(f, ic, p, dims, t_end; kwargs...)

    # chunks = Iterators.partition(ics, cld(length(ics), Threads.nthreads()))
    # tasks = map(chunks) do chunk
    #     Threads.@spawn map(eachindex(chunk)) do i
    #         @printf("%s, thread %2i, %5i\n", funcname, Threads.threadid(), i)
    #         ftle_sol(chunk[i])
    #     end
    # end
    # sols = mapreduce(fetch, vcat, tasks)

    tasks = map(eachindex(ics)) do i
        Threads.@spawn begin
            @printf("%s, thread %3i, %8i\n", funcname, Threads.threadid(), i)
            ftle_sol(ics[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end

