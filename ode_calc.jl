function calc_ode(f, ic, tspan, p; kwargs...)
    prob = DE.ODEProblem(f, ic, tspan, p)
    sol = DE.solve(prob; kwargs...).u |> stack
    return sol
end

function calc_ode_multiple_ic(f, ics, tspan, p; kwargs...)
    return map(ics) do ic
        prob = DE.ODEProblem(f, ic, tspan, p)
        DE.solve(prob; kwargs...).u |> stack
    end
end

function calc_ode_multiple_params(f, ic, tspan, ps; kwargs...)
    return map(ps) do p
        prob = DE.ODEProblem(f, ic, tspan, p)
        DE.solve(prob; kwargs...).u |> stack
    end
end
