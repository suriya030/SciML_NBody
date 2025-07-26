# scripts/03_train_neural_ode.jl

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization,
      OptimizationOptimisers, Random, JLD2

# Load the training data using a robust path
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")

# =============================================================================
# HYPERPARAMETERS - Easy to modify
# =============================================================================
num_trajectories = 5        # Number of trajectories to use for training
learning_rate = 0.001        # Optimizer step size
max_iterations = 100        # Number of training iterations
activation_function = tanh  # Activation function (tanh, relu, sigmoid, etc.)
# =============================================================================

# Use first trajectory as reference for setup
t_ref = Float32.(dataset[1].t)
u0_ref = Float32.(dataset[1].u0)

# 1. Define the neural network using Lux.jl
rng = Random.default_rng()
# dudt_nn = Lux.Chain(Lux.Dense(length(u0_ref), 64, activation_function),
#                     Lux.Dense(64, 64, activation_function),
#                     Lux.Dense(64, length(u0_ref)))

# Bigger network
dudt_nn = Lux.Chain(Lux.Dense(length(u0_ref), 128, activation_function),
                    Lux.Dense(128, 128, activation_function),
                    Lux.Dense(128, 64, activation_function),
                    Lux.Dense(64, length(u0_ref)))

# 2. Initialize the parameters (ps) and state (st) for the Lux model
p, st = Lux.setup(rng, dudt_nn)
ps = ComponentArray(p) # Wrap parameters in a ComponentArray

# 3. Define the Neural ODE
prob_neuralode = NeuralODE(dudt_nn, (t_ref[1], t_ref[end]), Tsit5();
                           saveat = t_ref)

# 4. Define the prediction function
function predict_neuralode(p)
    # Pass initial condition, parameters, and state to the NeuralODE problem
    Array(prob_neuralode(u0_ref, p, st)[1])
end

# 5. Define the loss function - MUST return only scalar loss
function loss_neuralode(p) # p = current NN parameters
    total_loss = 0.0
    for trajectory in dataset[1:num_trajectories]
        u0_traj = Float32.(trajectory.u0)
        data_traj = Float32.(trajectory.data)
        # Create temporary NeuralODE for this trajectory's initial condition
        pred = Array(prob_neuralode(u0_traj, p, st)[1])
        total_loss += sum(abs2, data_traj .- pred)
    end
    return total_loss / num_trajectories  # Average loss over selected trajectories
end

# 6. Set up the callback function to monitor training
# We'll compute predictions inside the callback for monitoring
callback = function (p, l)
    println("Loss: ", l)
    # Return false to continue training
    return false
end

# 7. Use Optimization.jl to set up and run the training
adtype = Optimization.AutoZygote() # Automatic differentiation backend that computes gradients of your loss function
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
# Wraps your loss_neuralode function so the optimizer can minimize it
optprob = Optimization.OptimizationProblem(optf, ps)
# Defines what to optimize (the loss) and what parameters to adjust (ps)

println("Starting training for the Neural ODE...")
result_neuralode = Optimization.solve(optprob,
                                        OptimizationOptimisers.Adam(learning_rate);
                                        callback = callback,
                                        maxiters = max_iterations)
println("Training complete.")

# The optimized parameters are in result_neuralode.u
ps_trained = result_neuralode.u

# Save the trained parameters
jldsave(joinpath(@__DIR__, "..", "data", "trained_neural_ode.jld2"); 
        ps_trained, st, dudt_nn)
println("Saved trained Neural ODE parameters to data/trained_neural_ode.jld2")

# Optional: Plot comparison between true and predicted trajectories
using Plots
pred_final = predict_neuralode(ps_trained)

# Plot all variables (using first trajectory)
data_plot = Float32.(dataset[1].data)

# Create 2x3 subplot layout with appropriate aspect ratio
plots_array = []

# Position plots (top row)
for body in 1:3
    idx_start = 3*(body-1) + 1
    p_pos = plot(t_ref, data_plot[idx_start, :], label="True x", linewidth=2, color=:red)
    plot!(p_pos, t_ref, pred_final[idx_start, :], label="Pred x", linestyle=:dash, linewidth=2, color=:red)
    plot!(p_pos, t_ref, data_plot[idx_start+1, :], label="True y", linewidth=2, color=:green)
    plot!(p_pos, t_ref, pred_final[idx_start+1, :], label="Pred y", linestyle=:dash, linewidth=2, color=:green)
    plot!(p_pos, t_ref, data_plot[idx_start+2, :], label="True z", linewidth=2, color=:blue)
    plot!(p_pos, t_ref, pred_final[idx_start+2, :], label="Pred z", linestyle=:dash, linewidth=2, color=:blue)
    title!(p_pos, "Body $body Positions")
    xlabel!(p_pos, "Time")
    ylabel!(p_pos, "Position")
    push!(plots_array, p_pos)
end

# Velocity plots (bottom row)
for body in 1:3
    idx_start = 9 + 3*(body-1) + 1
    p_vel = plot(t_ref, data_plot[idx_start, :], label="True vx", linewidth=2, color=:red)
    plot!(p_vel, t_ref, pred_final[idx_start, :], label="Pred vx", linestyle=:dash, linewidth=2, color=:red)
    plot!(p_vel, t_ref, data_plot[idx_start+1, :], label="True vy", linewidth=2, color=:green)
    plot!(p_vel, t_ref, pred_final[idx_start+1, :], label="Pred vy", linestyle=:dash, linewidth=2, color=:green)
    plot!(p_vel, t_ref, data_plot[idx_start+2, :], label="True vz", linewidth=2, color=:blue)
    plot!(p_vel, t_ref, pred_final[idx_start+2, :], label="Pred vz", linestyle=:dash, linewidth=2, color=:blue)
    title!(p_vel, "Body $body Velocities")
    xlabel!(p_vel, "Time")
    ylabel!(p_vel, "Velocity")
    push!(plots_array, p_vel)
end

# Combine all plots with appropriate aspect ratio
p_combined = plot(plots_array..., layout=(2,3), size=(1200, 800), 
                  plot_title="Neural ODE Training Results - All Variables")

# Save the plot
savefig(p_combined, joinpath(@__DIR__, "..", "plots", "neural_ode_training_results.png"))
display(p_combined)