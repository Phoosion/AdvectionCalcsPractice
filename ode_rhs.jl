include("./compute_rhs.jl")



function _point_vortex_rhs!(
    vortex_dx, vortex_dy,
    particle_dx, particle_dy,
    vortex_x, vortex_y,
    particle_x, particle_y,
    omega, alpha, beta
    # epsilon=0
)
    _compute_point_vortex_rhs!(
        vortex_dx, vortex_dy,
        vortex_x, vortex_y,
        omega,
        alpha, beta
    )
    _compute_particle_rhs!(
        particle_dx, particle_dy,
        particle_x, particle_y,
        vortex_x, vortex_y,
        omega,
        alpha, beta
    )
    return nothing
end

function _point_vortex_particle_tangent_rhs!(
    vortex_dx, vortex_dy,
    particle_dx, particle_dy,
    vortex_x, vortex_y,
    particle_x, particle_y,
    tangent_dx, tangent_dy,
    tangent_x, tangent_y,
    omega, alpha, beta,
    #  epsilon
)
    _compute_point_vortex_rhs!(
        vortex_dx, vortex_dy,
        vortex_x, vortex_y,
        omega,
        alpha, beta,
    )
    _compute_particle_and_tangent_rhs!(
        particle_dx, particle_dy,
        particle_x, particle_y,
        tangent_dx, tangent_dy,
        tangent_x, tangent_y,
        vortex_x, vortex_y,
        omega,
        alpha, beta,
    )

    return nothing
end


function point_vortex_ode!(du, u, p, t)
    (;
        omega,
        alpha, beta,
        nv, np,
        # epsilon
    ) = p

    vortex_x = @view u[1:2:2nv]
    vortex_y = @view u[2:2:2nv]
    vortex_dx = @view du[1:2:2nv]
    vortex_dy = @view du[2:2:2nv]
    particle_x = @view u[2nv+1:2:2(nv+np)]
    particle_y = @view u[2nv+2:2:2(nv+np)]
    particle_dx = @view du[2nv+1:2:2(nv+np)]
    particle_dy = @view du[2nv+2:2:2(nv+np)]

    _point_vortex_rhs!(
        vortex_dx, vortex_dy,
        particle_dx, particle_dy,
        vortex_x, vortex_y,
        particle_x, particle_y,
        omega,
        alpha, beta,
        # epsilon
    )

    return nothing
end


function point_vortex_ode(u, p, t)
    (;
        omega,
        alpha, beta,
        nv, np,
        # epsilon
    ) = p

    du = zeros(size(u))
    vortex_x = @view u[1:2:2nv]
    vortex_y = @view u[2:2:2nv]
    vortex_dx = @view du[1:2:2nv]
    vortex_dy = @view du[2:2:2nv]
    particle_x = @view u[2nv+1:2:2(nv+np)]
    particle_y = @view u[2nv+2:2:2(nv+np)]
    particle_dx = @view du[2nv+1:2:2(nv+np)]
    particle_dy = @view du[2nv+2:2:2(nv+np)]

    _point_vortex_rhs!(
        vortex_dx, vortex_dy,
        particle_dx, particle_dy,
        vortex_x, vortex_y,
        particle_x, particle_y,
        omega,
        alpha, beta,
        # epsilon
    )

    return du
end

function point_vortex_ode_ld!(du, u, p, t)

    (;
        omega,
        alpha, beta,
        gamma,
        nv, np,
        # epsilon
    ) = p

    vortex_x = @view u[1:2:2nv]
    vortex_y = @view u[2:2:2nv]
    vortex_dx = @view du[1:2:2nv]
    vortex_dy = @view du[2:2:2nv]
    particle_x = @view u[2nv+1:2:2(nv+np)]
    particle_y = @view u[2nv+2:2:2(nv+np)]
    particle_dx = @view du[2nv+1:2:2(nv+np)]
    particle_dy = @view du[2nv+2:2:2(nv+np)]
    d_ld = @view du[2(nv+np)+1:2(nv+np)+np]

    _point_vortex_rhs!(
        vortex_dx, vortex_dy,
        particle_dx, particle_dy,
        vortex_x, vortex_y,
        particle_x, particle_y,
        omega,
        alpha, beta,
        # epsilon
    )

    d_ld .= abs.(particle_dx) .^ gamma + abs.(particle_dy) .^ gamma

    return nothing
end


function point_vortex_ode_arclength!(du, u, p, t)
    (;
        omega,
        alpha, beta,
        nv, np,
        # epsilon
    ) = p
    vortex_x = @view u[1:2:2nv]
    vortex_y = @view u[2:2:2nv]
    vortex_dx = @view du[1:2:2nv]
    vortex_dy = @view du[2:2:2nv]
    particle_x = @view u[2nv+1:2:2(nv+np)]
    particle_y = @view u[2nv+2:2:2(nv+np)]
    particle_dx = @view du[2nv+1:2:2(nv+np)]
    particle_dy = @view du[2nv+2:2:2(nv+np)]
    d_arclength = @view du[2(nv+np)+1:2(nv+np)+np]

    _point_vortex_rhs!(
        vortex_dx, vortex_dy,
        particle_dx, particle_dy,
        vortex_x, vortex_y,
        particle_x, particle_y,
        omega,
        alpha, beta,
        # epsilon
    )
    d_arclength .= sqrt.(particle_dx .^ 2 + particle_dy .^ 2)
    return nothing
end

function point_vortex_ode_tangent!(du, u, p, t)
    (;
        omega,
        alpha, beta,
        nv, np,
        # epsilon
    ) = p

    n = nv + np
    vortex_x = @view u[1:2:2nv]
    vortex_y = @view u[2:2:2nv]
    vortex_dx = @view du[1:2:2nv]
    vortex_dy = @view du[2:2:2nv]
    particle_x = @view u[2nv+1:2:2(nv+np)]
    particle_y = @view u[2nv+2:2:2(nv+np)]
    particle_dx = @view du[2nv+1:2:2(nv+np)]
    particle_dy = @view du[2nv+2:2:2(nv+np)]
    tangent_x = @view u[2n+1:2:2n+4np]
    tangent_y = @view u[2n+2:2:2n+4np]
    tangent_dx = @view du[2n+1:2:2n+4np]
    tangent_dy = @view du[2n+2:2:2n+4np]

    _point_vortex_particle_tangent_rhs!(
        vortex_dx, vortex_dy,
        particle_dx, particle_dy,
        vortex_x, vortex_y,
        particle_x, particle_y,
        tangent_dx, tangent_dy,
        tangent_x, tangent_y,
        omega,
        alpha, beta,
        # epsilon
    )

    return nothing
end


function point_vortex_ode_particle_avg_distance!(du, u, p, t)
    (;
        omega,
        alpha, beta,
        nv, np
        # epsilon
    ) = p

    vortex_x = @view u[1:2:2nv]
    vortex_y = @view u[2:2:2nv]
    vortex_dx = @view du[1:2:2nv]
    vortex_dy = @view du[2:2:2nv]

    particle_x = @view u[2nv+1:2:2(nv+np)]
    particle_y = @view u[2nv+2:2:2(nv+np)]
    particle_dx = @view du[2nv+1:2:2(nv+np)]
    particle_dy = @view du[2nv+2:2:2(nv+np)]

    d_distance = @view du[2(nv+np)+1:2(nv+np)+np]

    particle_ic_x = @view u[2(nv+np)+np+1:2:2(nv+np)+np+2np]
    particle_ic_y = @view u[2(nv+np)+np+2:2:2(nv+np)+np+2np]

    _point_vortex_rhs!(
        vortex_dx, vortex_dy,
        particle_dx, particle_dy,
        vortex_x, vortex_y,
        particle_x, particle_y,
        omega, alpha, beta
    )

    d_distance .= sqrt.((particle_x - particle_ic_x) .^ 2 + (particle_y - particle_ic_y) .^ 2)

    return nothing
end
