function calc_ode(f, IC, tspan, p; kwargs...)
    prob = ODE.ODEProblem(f, IC, tspan, p)
    sol = ODE.solve(prob; kwargs...).u |> stack
    return sol
end

function calc_parallel_ode(f, ICs, tspan, p; kwargs...)
    funcname = StackTraces.stacktrace()[1].func |> string
    ode_sol = (IC) -> calc_ode(f, IC, tspan, p; kwargs...)

    # chunks = Iterators.partition(ICs, cld(length(ICs), Threads.nthreads()))
    # tasks = map(chunks) do chunk
    #     Threads.@spawn map(eachindex(chunk)) do i
    #         @printf("%s, thread %3i, %8i\n", funcname, Threads.threadid(), i)
    #         ode_sol(chunk[i])
    #     end
    # end
    # sol = mapreduce(fetch, vcat, tasks)

    tasks = map(eachindex(ICs)) do i
        Threads.@spawn begin
            @printf("%s, thread %3i, %8i\n", funcname, Threads.threadid(), i)
            ode_sol(ICs[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end

function calc_lyapunov_spectrum(f, IC, p, dims, T, N; kwargs...)
    kwargs = merge(
        NamedTuple(kwargs),
        (
            save_everystep=false,
            save_start=false,
            save_end=false,
        )
    )
    dt = T / N
    E = Matrix{Float64}(I, dims, dims) |> vec
    pertub_index = length(IC)+1:(length(IC)+dims^2)
    prob = ODE.ODEProblem(f, [IC; E], (0, T), p)
    integrator = ODE.init(prob; kwargs...)
    lambda = zeros(dims)
    for i = 1:N
        ODE.step!(integrator, dt, true)
        u = integrator.u
        M = reshape(u[pertub_index], dims, dims)
        Q, R = qr!(M)
        lambda += log.(abs.(diag(R)))
        u[pertub_index] = vec(Matrix(Q))
        ODE.set_u!(integrator, u)
    end
    return (IC[end-dims+1:end], lambda ./ T)
end

function calc_ftle(f, ic, p, dims, T; kwargs...)
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
    sol = calc_ode(f, ic_pertub, (0, T), p; kwargs...)
    M = reshape(sol[pertub_index], dims, dims)
    lambda = log(opnorm(M)) / T
    return (ic[end-dims+1:end], lambda)
end

function calc_parallel_lyapunov_spectrum(f, ICs, p, dims, T, N; kwargs...)
    funcname = StackTraces.stacktrace()[1].func |> string
    lyap_sol = (IC) -> calc_lyapunov_spectrum(f, IC, p, dims, T, N; kwargs...)

    # chunks = Iterators.partition(ICs, cld(length(ICs), Threads.nthreads()))
    # tasks = map(chunks) do chunk
    #     Threads.@spawn map(eachindex(chunk)) do i
    #         @printf("%s, thread %2i, %5i\n", funcname, Threads.threadid(), i)
    #         ftle_sol(chunk[i])
    #     end
    # end
    # sol = mapreduce(fetch, vcat, tasks)

    tasks = map(eachindex(ICs)) do i
        Threads.@spawn begin
            @printf("%s, thread %3i, %8i\n", funcname, Threads.threadid(), i)
            lyap_sol(ICs[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end


function calc_parallel_ftle(f, ICs, p, dims, T; kwargs...)
    funcname = StackTraces.stacktrace()[1].func |> string
    ftle_sol = (IC) -> calc_ftle(f, IC, p, dims, T; kwargs...)

    # chunks = Iterators.partition(ICs, cld(length(ICs), Threads.nthreads()))
    # tasks = map(chunks) do chunk
    #     Threads.@spawn map(eachindex(chunk)) do i
    #         @printf("%s, thread %2i, %5i\n", funcname, Threads.threadid(), i)
    #         ftle_sol(chunk[i])
    #     end
    # end
    # sols = mapreduce(fetch, vcat, tasks)

    tasks = map(eachindex(ICs)) do i
        Threads.@spawn begin
            @printf("%s, thread %3i, %8i\n", funcname, Threads.threadid(), i)
            ftle_sol(ICs[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end