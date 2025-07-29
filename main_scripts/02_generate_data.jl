# scripts/02_generate_data.jl : Checked

using JLD2
using LinearAlgebra
using Random
include(joinpath(@__DIR__, "..", "src", "nbody.jl"))

function generate_trajectory(u0, tspan, p; noise_level=0.01, saveat=0.1)
    prob = ODEProblem(n_body_system!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=saveat)
    
    # Add Gaussian noise to the solution to simulate measurement error
    noisy_data = Array(sol) + noise_level * randn(size(Array(sol)))
    
    # sol.u[1...18] # 18 variables p1x,p1y,p1z ... v3x,v3y,v3z
    # size( sol.u[1] ) = 101 i.e u[1] = p1x(t) sampled at 101 timesteps
    return sol.t, noisy_data
end

# Set seed for reproducibility
Random.seed!(42)

# --- Main Data Generation ---
# Base parameters
p = Float32[1.0, 1.0, 1.0, 1.0]
tspan = (0.0f0, 10.0f0)

# Generate and save a dataset of 100 trajectories
dataset = []
for i in 1:100
    # Generate different initial conditions for each trajectory
    # Random positions 
    positions = Float32.(1 * randn(9))  # 3 bodies × 3 coordinates (x,y,z each)
    # Random velocities 
    velocities = Float32.(1 * randn(9))  # 3 bodies × 3 velocity components
    
    u0_initial = vcat(positions, velocities)
    
    t, data = generate_trajectory(u0_initial, tspan, p, noise_level=0.0)
    push!(dataset, (t=t, data=data, u0=u0_initial))
end

# Save the entire dataset to a single file
jldsave(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"); dataset)
println("Saved dataset with $(length(dataset)) trajectories to data/nbody_dataset.jld2")