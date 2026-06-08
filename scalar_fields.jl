function particle_hamiltonian(
    xi, eta,
    x, y,
    omega,
    alpha, beta
)
    @assert length(omega) == length(x) == length(y)
    vortex_part = (
        omega' * log.(
            (xi .- x) .^ 2 .+ (eta .- y) .^ 2
        )
    ) * (1 / 4 / pi)
    flow_part = (alpha + beta)xi^2 / 2 + (alpha - beta)eta^2 / 2
    phi = -vortex_part + flow_part
    return phi
end

function particle_velocity(
    xi, eta,
    x, y,
    omega,
    alpha, beta
)
    @assert length(omega) == length(x) == length(y)
    R = 1 / 2 / pi
    delta_y = eta .- y
    delta_x = xi .- x
    inv_r_sqr = 1 / (delta_x .^ 2 .+ delta_y .^ 2)
    flow_x = (alpha - beta) * eta
    flow_y = -(alpha + beta) * xi
    d_xi = -R .* (delta_y .* inv_r_sqr) * omega .+ flow_x
    d_eta = R .* (delta_x .* inv_r_sqr) * omega .+ flow_y
    return d_xi, d_eta
end
