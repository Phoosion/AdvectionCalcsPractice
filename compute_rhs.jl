@inline function _compute_point_vortex_rhs!(
    vortex_dx, vortex_dy,
    vortex_x, vortex_y,
    omega,
    alpha, beta
)
    R = 1 / 2 / pi
    delta_x = vortex_x .- vortex_x'
    delta_y = vortex_y .- vortex_y'
    inv_r_sqr = 1 ./ (delta_x .^ 2 .+ delta_y .^ 2)
    inv_r_sqr[inv_r_sqr.==Inf] .= 0
    flow_x = (alpha - beta) .* vortex_y
    flow_y = -(alpha + beta) .* vortex_x
    vortex_dx .= -R .* (delta_y .* inv_r_sqr) * omega .+ flow_x
    vortex_dy .= R .* (delta_x .* inv_r_sqr) * omega .+ flow_y
end


@inline function _compute_particle_rhs!(
    particle_dx, particle_dy,
    particle_x, particle_y,
    vortex_x, vortex_y,
    omega,
    alpha, beta
)
    R = 1 / 2 / pi
    delta_x = particle_x .- vortex_x'
    delta_y = particle_y .- vortex_y'
    inv_r_sqr = 1 ./ (delta_x .^ 2 .+ delta_y .^ 2)
    flow_x = (alpha - beta) .* particle_y
    flow_y = -(alpha + beta) .* particle_x
    particle_dx .= -R .* (delta_y .* inv_r_sqr) * omega .+ flow_x
    particle_dy .= R .* (delta_x .* inv_r_sqr) * omega .+ flow_y
end

@inline function _compute_tangent_rhs!(
    tangent_dx, tangent_dy,
    tangent_x, tangent_y,
    particle_dxdx, particle_dxdy,
    particle_dydx, particle_dydy
)
    tangent_index = collect(eachindex(tangent_dx))
    particle_index = div.(tangent_index .- 1, 2) .+ 1
    tangent_dx .= particle_dxdx[particle_index] .* tangent_x + particle_dxdy[particle_index] .* tangent_y
    tangent_dy .= particle_dydx[particle_index] .* tangent_x + particle_dydy[particle_index] .* tangent_y
end

@inline function _compute_particle_and_tangent_rhs!(
    particle_dx, particle_dy,
    particle_x, particle_y,
    tangent_dx, tangent_dy,
    tangent_x, tangent_y,
    vortex_x, vortex_y,
    omega,
    alpha, beta
)
    R = 1 / 2 / pi
    delta_x = particle_x .- vortex_x'
    delta_y = particle_y .- vortex_y'
    inv_r_sqr = 1 ./ (delta_x .^ 2 .+ delta_y .^ 2)

    flow_x = (alpha - beta) .* particle_y
    flow_y = -(alpha + beta) .* particle_x
    flow_x_dy = (alpha - beta)
    flow_y_dx = -(alpha + beta)

    particle_dxdx = 2R .* (delta_x .* delta_y .* inv_r_sqr .^ 2) * omega
    particle_dydy = -particle_dxdx
    particle_dxdy = -R .* (inv_r_sqr .* (1 .- 2 .* delta_y .^ 2 .* inv_r_sqr)) * omega .+ flow_x_dy
    particle_dydx = R .* (inv_r_sqr .* (1 .- 2 .* delta_x .^ 2 .* inv_r_sqr)) * omega .+ flow_y_dx

    particle_dx .= -R .* (delta_y .* inv_r_sqr) * omega .+ flow_x
    particle_dy .= R .* (delta_x .* inv_r_sqr) * omega .+ flow_y

    _compute_tangent_rhs!(
        tangent_dx, tangent_dy,
        tangent_x, tangent_y,
        particle_dxdx, particle_dxdy,
        particle_dydx, particle_dydy
    )
end