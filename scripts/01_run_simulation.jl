# scripts/01_run_simulation.jl

using Plots
include(joinpath(@__DIR__, "..", "src", "nbody.jl"))

# Initial conditions for a stable 3-body system
u0 = [
    # Positions (x, y, z)
    1.0, 0.0, 0.0, -0.5, 0.866, 0.0, -0.5, -0.866, 0.0,
    # Velocities (vx, vy, vz)
    0.0, 0.5, 0.0, -0.433, -0.25, 0.0, 0.433, -0.25, 0.0
]

# Parameters: [m1, m2, m3, G]
p = [1.0, 1.0, 1.0, 1.0]
tspan = (0.0, 10.0)

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