function calc_method_in_parallel(method, ics)
    funcname = join(getproperty.(StackTraces.stacktrace()[1:2], :func), " ")
    tasks = map(eachindex(ics)) do i
        Threads.@spawn begin
            Printf.@printf("%s || thread %3i, %8i\n", funcname, Threads.threadid(), i)
            method(ics[i])
        end
    end
    sol = map(fetch, tasks)
    return sol
end

function calc_parallel_ode(f, ics, tspan, p; kwargs...)
    ode_sol = (ic) -> calc_ode(f, ic, tspan, p; kwargs...)
    return calc_method_in_parallel(ode_sol, ics)
end

function calc_parallel_ftle(f, ics, p, dims, t_end; kwargs...)
    ftle_sol = (ic) -> calc_ftle(f, ic, p, dims, t_end; kwargs...)
    return calc_method_in_parallel(ftle_sol, ics)
end

function calc_parallel_finite_diff_ftle(f, ics, p, t_end; sol_axis=nothing, kwargs...)
    ftle_sol = (ic) -> calc_finite_diff_ftle(f, ic, p, t_end; sol_axis=sol_axis, kwargs...)
    return calc_method_in_parallel(ftle_sol, ics)
end

function calc_parallel_poincare_map(f, ics, p, hyperplane, n_points; t_max=1e6, kwargs...)
    poincare_map_sol = (ic) -> calc_poincare_map(f, ic, p, hyperplane, n_points; t_max=t_max, kwargs...)
    return calc_method_in_parallel(poincare_map_sol, ics)
end
