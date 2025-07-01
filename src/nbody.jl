using DifferentialEquations

"""
    n_body_system!(du, u, p, t)

In-place function for the gravitational N-body problem.
- `u`: State vector [r1_x, r1_y, r1_z, ..., v1_x, v1_y, v1_z, ...]
- `p`: Parameter vector [m1, m2, ..., G]
"""
function n_body_system!(du, u, p, t)
    N = length(p) - 1 # Number of bodies
    G = p[end]
    
    # Create views for positions (r) and velocities (v)
    r = @view u[1:3*N]
    v = @view u[3*N+1:end]
    
    # The derivative of position is velocity
    dr = @view du[1:3*N]
    dr .= v
    
    # The derivative of velocity is acceleration
    dv = @view du[3*N+1:end]
    dv .= 0.0 # Initialize accelerations

    # Calculate gravitational accelerations by summing forces
    for i in 1:N
        for j in (i+1):N
            ri = @view r[3*(i-1)+1 : 3*i]
            rj = @view r[3*(j-1)+1 : 3*j]
            
            r_ij = rj - ri
            dist_cubed = sum(r_ij.^2)^(3/2)
            
            force_ij = G * r_ij / dist_cubed
            
            # Update accelerations according to F=ma (a = F/m)
            # The mass of the body is incorporated in the parameters
            dv[3*(i-1)+1 : 3*i] .+= p[j] * force_ij
            dv[3*(j-1)+1 : 3*j] .-= p[i] * force_ij
        end
    end
end