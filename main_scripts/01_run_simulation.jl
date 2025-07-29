# scripts/01_run_simulation.jl : Checked

using Plots
include(joinpath(@__DIR__, "..", "src", "nbody.jl"))

# Initial conditions for a stable 3-body system
u0 = Float32[
    # Positions (x, y, z)
    1.0, 0.0, 0.0, -0.5, 0.866, 0.0, -0.5, -0.866, 0.0,
    # Velocities (vx, vy, vz)
    0.0, 0.5, 0.0, -0.433, -0.25, 0.0, 0.433, -0.25, 0.0
]

# Parameters: [m1, m2, m3, G]
p = Float32[1.0, 1.0, 1.0, 1.0]
tspan = (0.0f0, 10.0f0)

# Create and solve the ODE problem
prob = ODEProblem(n_body_system!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.01)

# Plot the trajectories and save the figure
p1 = plot(sol, idxs=(1, 2), label="Body 1", title="3-Body Problem Simulation")
plot!(p1, sol, idxs=(4, 5), label="Body 2")
plot!(p1, sol, idxs=(7, 8), label="Body 3")

# Save the plot
savefig(p1, joinpath(@__DIR__, "..", "plots", "classical_simulation.png"))
display(p1) # Show plot if running interactively

# Plot the 3D trajectories and save the figure
p1_3d = plot(sol, idxs=(1, 2, 3), label="Body 1", title="3D 3-Body Problem")
plot!(p1_3d, sol, idxs=(4, 5, 6), label="Body 2")
plot!(p1_3d, sol, idxs=(7, 8, 9), label="Body 3")

# Save the 3D plot
savefig(p1_3d, joinpath(@__DIR__, "..", "plots", "3d_3body_simulation.png"))
display(p1_3d) # Show the 3D plot if running interactively