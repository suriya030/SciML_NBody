using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization,
      OptimizationOptimisers, Random, JLD2, LuxCUDA, Plots

# =============================================================================
# GPU SETUP
# =============================================================================
# Set up the GPU device. This will use the GPU if available.
# If no GPU is available, it will default to the CPU.
const gdev = gpu_device(1)
const cdev = cpu_device()
println("Training will run on: ", gdev)
# =============================================================================

# Load the training data using a robust path
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")

# =============================================================================
# HYPERPARAMETERS - Easy to modify
# =============================================================================
num_trajectories = 1      # Number of trajectories to use for training
learning_rate = 0.001     # Optimizer step size
max_iterations = 10000    # Number of training iterations
activation_function = tanh # Activation function (tanh, relu, sigmoid, etc.)
# =============================================================================

# Use first trajectory as reference for setup
t_ref = Float32.(dataset[1].t)
u0_ref = Float32.(dataset[1].u0) # This remains on the CPU for reference

# 1. Define the neural network using Lux.jl
rng = Random.default_rng()
# Bigger network
dudt_nn = Lux.Chain(Lux.Dense(length(u0_ref), 128, activation_function),
                      Lux.Dense(128, 128, activation_function),
                      Lux.Dense(128, 64, activation_function),
                      Lux.Dense(64, length(u0_ref)))

# 2. Initialize the parameters (ps) and state (st) and move them to the GPU
p, st = Lux.setup(rng, dudt_nn)
ps = ComponentArray(p) |> gdev # Move initial parameters to the GPU
st = st |> gdev                 # Move state to the GPU

# 3. Define the Neural ODE
# The solver will run on the GPU as long as u0 and p are on the GPU
prob_neuralode = NeuralODE(dudt_nn, (t_ref[1], t_ref[end]), Tsit5();
                           saveat = t_ref)

# 4. Define the prediction function for validation/plotting
function predict_neuralode(p)
    # Move the reference initial condition to the GPU for this prediction
    u0_gpu = u0_ref |> gdev
    # The prediction runs on the GPU
    pred_gpu = prob_neuralode(u0_gpu, p, st)[1]
    # Move the result back to the CPU for plotting
    return Array(pred_gpu)
end

# 5. Define the loss function - MUST return only scalar loss
function loss_neuralode(p) # p = current NN parameters (on GPU)
    total_loss = 0.0f0
    for trajectory in dataset[1:num_trajectories]
        # Move the trajectory's initial condition and data to the GPU
        u0_traj = Float32.(trajectory.u0) |> gdev
        data_traj = Float32.(trajectory.data) |> gdev

        # The prediction is computed on the GPU
        pred = Array(prob_neuralode(u0_traj, p, st)[1]) 

        # The loss is calculated on the GPU, resulting in a scalar
        total_loss += sum(abs2, data_traj .- pred)
    end
    return total_loss / num_trajectories # Average loss
end

# 6. Set up the callback function to monitor training
callback = function (p, l)
    println("Loss: ", l)
    # Return false to continue training
    return false
end

# 7. Use Optimization.jl to set up and run the training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
# The parameters `ps` are already on the GPU
optprob = Optimization.OptimizationProblem(optf, ps)

println("Starting training for the Neural ODE...")
result_neuralode = Optimization.solve(optprob,
                                       OptimizationOptimisers.Adam(learning_rate);
                                       callback = callback,
                                       maxiters = max_iterations)
println("Training complete.")

# The optimized parameters are on the GPU
ps_trained_gpu = result_neuralode.u

# Move the trained parameters and state back to the CPU for saving
ps_trained_cpu = ps_trained_gpu |> cdev
st_cpu = st |> cdev

# Save the trained parameters
jldsave(joinpath(@__DIR__, "..", "data", "trained_neural_ode.jld2");
        ps_trained = ps_trained_cpu, st = st_cpu, dudt_nn)
println("Saved trained Neural ODE parameters to data/trained_neural_ode.jld2")

# Optional: Plot comparison between true and predicted trajectories
# The predict_neuralode function already moves the result to the CPU
pred_final = predict_neuralode(ps_trained_gpu)

# Plot all variables (using first trajectory)
data_plot = Float32.(dataset[1].data) # This data is on the CPU

# Create 2x3 subplot layout
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

# Combine all plots
p_combined = plot(plots_array..., layout=(2,3), size=(1200, 800),
                  plot_title="Neural ODE Training Results - All Variables")

# Save the plot
savefig(p_combined, joinpath(@__DIR__, "..", "plots", "neural_ode_training_results_gpu.png"))
display(p_combined)
