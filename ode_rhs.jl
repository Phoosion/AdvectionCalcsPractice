function _point_vortex_rhs!(
    dx, dy,
    x, y,
    nv, np,
    omega, alpha, beta
)
    R = 1 / 2 / pi
    delta_y = [y[i] - y[j] for i = 1:(nv+np), j = 1:nv]
    delta_x = [x[i] - x[j] for i = 1:(nv+np), j = 1:nv]
    r = [1 / (delta_x[i, j]^2 + delta_y[i, j]^2) for i = 1:(nv+np), j = 1:nv]
    flow_x = (alpha - beta) * y
    flow_y = -(alpha + beta) * x

    @inbounds @inline vortex_dx(i) = -R * sum(omega[j] * delta_y[i, j] * r[i, j] for j = [1:i-1; i+1:nv]; init=0) + flow_x[i]
    @inbounds @inline vortex_dy(i) = R * sum(omega[j] * delta_x[i, j] * r[i, j] for j = [1:i-1; i+1:nv]; init=0) + flow_y[i]
    @inbounds @inline particle_dx(i) = -R * sum(omega[j] * delta_y[i, j] * r[i, j] for j = 1:nv; init=0) + flow_x[i]
    @inbounds @inline particle_dy(i) = R * sum(omega[j] * delta_x[i, j] * r[i, j] for j = 1:nv; init=0) + flow_y[i]

    # @inbounds @inline delta_y(i, j) = (y[i] - y[j])
    # @inbounds @inline delta_x(i, j) = (x[i] - x[j])

    # @inbounds @inline r(i, j) = 1 / (delta_x(i, j)^2 + delta_y(i, j)^2)

    # @inbounds @inline flow_x(i) = (alpha - beta)y[i]
    # @inbounds @inline flow_y(i) = -(alpha + beta)x[i]

    # @inbounds @inline vortex_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_x(i)
    # @inbounds @inline vortex_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = [1:i-1; i+1:nv]; init=0) + flow_y(i)

    # @inbounds @inline particle_dx(i) = -R * sum(omega[j] * delta_y(i, j) * r(i, j) for j = 1:nv; init=0) + flow_x(i)
    # @inbounds @inline particle_dy(i) = R * sum(omega[j] * delta_x(i, j) * r(i, j) for j = 1:nv; init=0) + flow_y(i)

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

function _point_vortex_particle_tangent_rhs!(
    dtangent_x, dtangent_y,
    tangent_x, tangent_y,
    x, y,
    nv, np, n_tangent,
    omega, alpha, beta
)
    R = 1 / 2 / pi
    delta_y = [y[i] - y[j] for i = 1:(nv+np), j = 1:nv]
    delta_x = [x[i] - x[j] for i = 1:(nv+np), j = 1:nv]
    r = [1 / (delta_x[i, j]^2 + delta_y[i, j]^2) for i = 1:(nv+np), j = 1:nv]
    flow_x_dy = (alpha - beta)
    flow_y_dx = -(alpha + beta)

    @inbounds @inline particle_dxdx(i) = 2R * sum(omega[j] * delta_y[i, j] * delta_x[i, j] * r[i, j]^2 for j = 1:nv; init=0)
    @inbounds @inline particle_dxdy(i) = -R * sum(omega[j] * r[i, j] * (1 - 2 * delta_y[i, j]^2 * r[i, j]) for j = 1:nv; init=0) + flow_x_dy
    @inbounds @inline particle_dydx(i) = R * sum(omega[j] * r[i, j] * (1 - 2 * delta_x[i, j]^2 * r[i, j]) for j = 1:nv; init=0) + flow_y_dx
    @inbounds @inline particle_dydy(i) = -2R * sum(omega[j] * delta_y[i, j] * delta_x[i, j] * r[i, j]^2 for j = 1:nv; init=0)
    @inbounds @inline tangent_dx(i, j) = particle_dxdx(i) * tangent_x[j] + particle_dxdy(i) * tangent_y[j]
    @inbounds @inline tangent_dy(i, j) = particle_dydx(i) * tangent_x[j] + particle_dydy(i) * tangent_y[j]

    # @inbounds @inline delta_y(i, j) = (y[i] - y[j])
    # @inbounds @inline delta_x(i, j) = (x[i] - x[j])
    # @inbounds @inline r(i, j) = 1 / (delta_x(i, j)^2 + delta_y(i, j)^2)

    # @inbounds @inline particle_dxdx(i) = 2R * sum(omega[j] * delta_y(i, j) * delta_x(i, j) * r(i, j)^2 for j = 1:nv; init=0)
    # @inbounds @inline particle_dxdy(i) = -R * sum(omega[j] * r(i, j) * (1 - 2 * delta_y(i, j)^2 * r(i, j)) for j = 1:nv; init=0) + flow_x_dy
    # @inbounds @inline particle_dydx(i) = R * sum(omega[j] * r(i, j) * (1 - 2 * delta_x(i, j)^2 * r(i, j)) for j = 1:nv; init=0) + flow_y_dx
    # @inbounds @inline particle_dydy(i) = -2R * sum(omega[j] * delta_y(i, j) * delta_x(i, j) * r(i, j)^2 for j = 1:nv; init=0)

    # @inbounds @inline tangent_dx(i, j) = particle_dxdx(i) * x[j] + particle_dxdy(i) * y[j]
    # @inbounds @inline tangent_dy(i, j) = particle_dydx(i) * x[j] + particle_dydy(i) * y[j]

    @inbounds for k = (1:n_tangent)
        i = div(k - 1, 2) + 1 + nv
        dtangent_x[k] = tangent_dx(i, k)
        dtangent_y[k] = tangent_dy(i, k)
        # i = k + nv
        # j = 2k + nv + np - 1
        # dx[j] = tangent_dx(i, j)
        # dy[j] = tangent_dy(i, j)
        # dx[j+1] = tangent_dx(i, j + 1)
        # dy[j+1] = tangent_dy(i, j + 1)
    end

    return nothing
end


function point_vortex_ode!(du, u, p, t)
    (; omega, alpha, beta, nv, np) = p

    x = @view u[1:2:end]
    y = @view u[2:2:end]
    dx = @view du[1:2:end]
    dy = @view du[2:2:end]

    _point_vortex_rhs!(dx, dy, x, y, nv, np, omega, alpha, beta)

    return nothing
end

function point_vortex_ode_ld!(du, u, p, t)
    (; omega, alpha, beta, gamma, nv, np) = p

    x = @view u[1:2:2(nv+np)]
    y = @view u[2:2:2(nv+np)]
    dx = @view du[1:2:2(nv+np)]
    dy = @view du[2:2:2(nv+np)]
    dld = @view du[2(nv+np)+1:2(nv+np)+np]

    @inbounds @inline ld(i) = abs(dx[i])^gamma + abs(dy[i])^gamma

    _point_vortex_rhs!(dx, dy, x, y, nv, np, omega, alpha, beta)

    @inbounds for k = 1:np
        dld[k] = ld(k + nv)
    end

    return nothing
end


function point_vortex_ode_arclength!(du, u, p, t)
    (; omega, alpha, beta, nv, np) = p

    x = @view u[1:2:2(nv+np)]
    y = @view u[2:2:2(nv+np)]
    dx = @view du[1:2:2(nv+np)]
    dy = @view du[2:2:2(nv+np)]
    d_arclength = @view du[2(nv+np)+1:2(nv+np)+np]

    @inline @inbounds arclength(i) = sqrt(dx[i]^2 + dy[i]^2)

    _point_vortex_rhs!(dx, dy, x, y, nv, np, omega, alpha, beta)

    @inbounds for k = 1:np
        d_arclength[k] = arclength(k + nv)
    end

    return nothing
end

function point_vortex_ode_tangent!(du, u, p, t)
    (; omega, alpha, beta, nv, np) = p
    x = @view u[1:2:2(nv+np)]
    y = @view u[2:2:2(nv+np)]
    dx = @view du[1:2:2(nv+np)]
    dy = @view du[2:2:2(nv+np)]

    tangent_x = @view u[2(nv+np)+1:2:2(nv+np)+2np]
    tangent_y = @view u[2(nv+np)+2:2:2(nv+np)+2np]
    dtangent_x = @view du[2(nv+np)+1:2:2(nv+np)+2np]
    dtangent_y = @view du[2(nv+np)+2:2:2(nv+np)+2np]

    _point_vortex_rhs!(dx, dy, x, y, nv, np, omega, alpha, beta)
    _point_vortex_particle_tangent_rhs!(dtangent_x, dtangent_y, tangent_x, tangent_y, x, y, nv, np, 2np, omega, alpha, beta)

    return nothing
end

function point_vortex_ode_combined!(du, u, p, t)
    (; omega, alpha, beta, nv, np, n_tangents, n_arclengths) = p
    n = np + nv
    x = @view u[1:2:2n]
    y = @view u[2:2:2n]
    dx = @view du[1:2:2n]
    dy = @view du[2:2:2n]

    tangent_x = @view u[2n+1:2:2(n+n_tangents)]
    tangent_y = @view u[2n+2:2:2(n+n_tangents)]

    dtangent_x = @view du[2n+1:2:2(n+n_tangents)]
    dtangent_y = @view du[2n+2:2:2(n+n_tangents)]

    # d_ld = @view u[2(n+n_tangents)+1:2(n+n_tangents)+n_lds]
    d_arclength = @view u[2(n+n_tangents)+1:2(n+n_tangents)+n_arclengths]

    # @inbounds @inline ld(i) = abs(dx[i])^gamma + abs(dy[i])^gamma
    @inbounds @inline arclength(i) = sqrt(dx[i]^2 + dy[i]^2)

    _point_vortex_rhs!(dx, dy, x, y, nv, np, omega, alpha, beta)
    _point_vortex_particle_tangent_rhs!(dtangent_x, dtangent_y, tangent_x, tangent_y, x, y, nv, np, n_tangents, omega, alpha, beta)

    @inbounds for k = (1:n_arclengths)
        d_arclength[k] = arclength(k + nv)
    end

    return nothing
end