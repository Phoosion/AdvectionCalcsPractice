

function point_vortex_ode!(du, u, p, t)
    omega, alpha, beta = p
    nv = length(omega)
    np = div(length(u), 2) - nv
    R = 1 / 2 / pi
    x = @view u[1:2:end]
    y = @view u[2:2:end]
    dx = @view du[1:2:end]
    dy = @view du[2:2:end]

    @inbounds @inline delta_y(i, j) = (y[i] - y[j])
    @inbounds @inline delta_x(i, j) = (x[i] - x[j])

    @inbounds @inline r(i, j) = 1 / (delta_x(i, j)^2 + delta_y(i, j)^2)

    @inbounds @inline flow_x(i) = (alpha - beta)y[i]
    @inbounds @inline flow_y(i) = -(alpha + beta)x[i]

    @inbounds @inline vortex_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_x(i)
    @inbounds @inline vortex_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_y(i)

    @inbounds @inline particle_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = 1:nv; init=0) + flow_x(i)
    @inbounds @inline particle_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = 1:nv; init=0) + flow_y(i)

    @inbounds for i = 1:nv
        dx[i] = vortex_dx(i)
        dy[i] = vortex_dy(i)
    end
    @inbounds for k = (1:np) .+ nv
        dx[k] = particle_dx(k)
        dy[k] = particle_dy(k)
    end
    return nothing
end

function point_vortex_ode_ld!(du, u, p, t)
    omega, alpha, beta, gamma = p
    nv = length(omega)
    R = 1 / 2 / pi
    x = @view u[1:2:end-1]
    y = @view u[2:2:end-1]
    dx = @view du[1:2:end-1]
    dy = @view du[2:2:end-1]

    @inbounds @inline delta_y(i, j) = (y[i] - y[j])
    @inbounds @inline delta_x(i, j) = (x[i] - x[j])

    @inbounds @inline r(i, j) = 1 / (delta_x(i, j)^2 + delta_y(i, j)^2)

    @inbounds @inline flow_x(i) = (alpha - beta)y[i]
    @inbounds @inline flow_y(i) = -(alpha + beta)x[i]

    @inbounds @inline vortex_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_x(i)
    @inbounds @inline vortex_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_y(i)

    @inbounds @inline particle_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = 1:nv; init=0) + flow_x(i)
    @inbounds @inline particle_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = 1:nv; init=0) + flow_y(i)
    # @inbounds @inline ld(gamma) = abs(dx[nv+1])^gamma + abs(dy[nv+1])^gamma

    @inbounds for i = 1:nv
        dx[i] = vortex_dx(i)
        dy[i] = vortex_dy(i)
    end

    dx[nv+1] = particle_dx(nv + 1)
    dy[nv+1] = particle_dy(nv + 1)

    du[end] = abs(dx[nv+1])^gamma + abs(dy[nv+1])^gamma
    return nothing
end


function point_vortex_ode_arclength!(du, u, p, t)
    omega, alpha, beta = p

    nv = length(omega)
    R = 1 / 2 / pi

    x = @view u[1:2:end-1]
    y = @view u[2:2:end-1]
    dx = @view du[1:2:end-1]
    dy = @view du[2:2:end-1]

    @inbounds @inline delta_y(i, j) = (y[i] - y[j])
    @inbounds @inline delta_x(i, j) = (x[i] - x[j])
    @inbounds @inline r(i, j) = 1 / (delta_x(i, j)^2 + delta_y(i, j)^2)

    @inbounds @inline flow_x(i) = (alpha - beta)y[i]
    @inbounds @inline flow_y(i) = -(alpha + beta)x[i]

    @inbounds @inline vortex_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_x(i)
    @inbounds @inline vortex_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_y(i)

    @inbounds @inline particle_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = 1:nv; init=0) + flow_x(i)
    @inbounds @inline particle_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = 1:nv; init=0) + flow_y(i)

    @inbounds for i = 1:nv
        dx[i] = vortex_dx(i)
        dy[i] = vortex_dy(i)
    end

    dx[nv+1] = particle_dx(nv + 1)
    dy[nv+1] = particle_dy(nv + 1)

    du[end] = sqrt(dx[nv+1]^2 + dy[nv+1]^2)
    return nothing
end


function point_vortex_ode_tangent!(du, u, p, t)
    omega, alpha, beta = p
    nv = length(omega)
    R = 1 / 2 / pi

    x = @view u[1:2:end]
    y = @view u[2:2:end]
    dx = @view du[1:2:end]
    dy = @view du[2:2:end]

    @inbounds @inline delta_y(i, j) = (y[i] - y[j])
    @inbounds @inline delta_x(i, j) = (x[i] - x[j])
    @inbounds @inline r(i, j) = 1 / (delta_x(i, j)^2 + delta_y(i, j)^2)

    @inbounds @inline flow_x(i) = (alpha - beta)y[i]
    @inbounds @inline flow_y(i) = -(alpha + beta)x[i]

    flow_x_dy = (alpha - beta)
    flow_y_dx = -(alpha + beta)

    @inbounds @inline vortex_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_x(i)
    @inbounds @inline vortex_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_y(i)

    @inbounds @inline particle_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = 1:nv; init=0) + flow_x(i)
    @inbounds @inline particle_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = 1:nv; init=0) + flow_y(i)

    @inbounds @inline particle_dxdx(i) = 2R * sum(omega[j] * delta_y(i, j) * delta_x(i, j) * r(i, j)^2 for j = 1:nv; init=0)
    @inbounds @inline particle_dxdy(i) = -R * sum(omega[j] * r(i, j) * (1 - 2 * delta_y(i, j)^2 * r(i, j)) for j = 1:nv; init=0) + flow_x_dy
    @inbounds @inline particle_dydx(i) = R * sum(omega[j] * r(i, j) * (1 - 2 * delta_x(i, j)^2 * r(i, j)) for j = 1:nv; init=0) + flow_y_dx
    @inbounds @inline particle_dydy(i) = -2R * sum(omega[j] * delta_y(i, j) * delta_x(i, j) * r(i, j)^2 for j = 1:nv; init=0)

    @inbounds @inline tangent_dx(i, j) = particle_dxdx(i) * x[j] + particle_dxdy(i) * y[j]
    @inbounds @inline tangent_dy(i, j) = particle_dydx(i) * x[j] + particle_dydy(i) * y[j]

    @inbounds for i = 1:nv
        dx[i] = vortex_dx(i)
        dy[i] = vortex_dy(i)
    end

    dx[nv+1] = particle_dx(nv + 1)
    dy[nv+1] = particle_dy(nv + 1)

    dx[nv+2] = tangent_dx(nv + 1, nv + 2)
    dy[nv+2] = tangent_dy(nv + 1, nv + 2)

    dx[nv+3] = tangent_dx(nv + 1, nv + 3)
    dy[nv+3] = tangent_dy(nv + 1, nv + 3)

    return nothing
end