# scripts/02_generate_data.jl

using JLD2
using LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "nbody.jl"))

function generate_trajectory(u0, tspan, p; noise_level=0.01, saveat=0.1)
    prob = ODEProblem(n_body_system!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=saveat)
    
    # Add Gaussian noise to the solution to simulate measurement error
    noisy_data = Array(sol) + noise_level * randn(size(Array(sol)))
    return sol.t, noisy_data
end

# --- Main Data Generation ---
# Base initial conditions and parameters
u0 = [1.0, 0.0, 0.0, -0.5, 0.866, 0.0, -0.5, -0.866, 0.0,
      0.0, 0.5, 0.0, -0.433, -0.25, 0.0, 0.433, -0.25, 0.0]
p = [1.0, 1.0, 1.0, 1.0]
tspan = (0.0, 10.0)

# Generate and save a dataset of 10 trajectories
dataset = []
for i in 1:100
    # Perturb initial positions slightly for variety
    u0_perturbed = u0 .+ 0.1 * randn(length(u0))
    t, data = generate_trajectory(u0_perturbed, tspan, p, noise_level=0.02)
    push!(dataset, (t=t, data=data, u0=u0_perturbed))
end

# Save the entire dataset to a single file
jldsave(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"); dataset)
println("Saved dataset with $(length(dataset)) trajectories to data/nbody_dataset.jld2")