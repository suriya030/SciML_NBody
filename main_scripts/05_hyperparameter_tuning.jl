# scripts/04_hyperparameter_tuning.jl

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization,
      OptimizationOptimisers, Random, JLD2, Plots, DataFrames, CSV

# Load the training data
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")

# =============================================================================
# HYPERPARAMETER CONFIGURATIONS
# =============================================================================
# Define all hyperparameter combinations to test
configs = []

# Activation functions to test
activation_functions = [tanh, sigmoid, relu]
activation_names = ["tanh", "sigmoid", "relu"]

# Network architectures (input_size -> hidden_layers -> output_size)
# Format: [(layer1_size, layer2_size, ...), name]
architectures = [
    # Shallow networks (1-2 layers)
    ([32], "tiny_1layer"),                    # 1. Very small: 1 layer with 32 neurons
    ([64], "small_1layer"),                   # 2. Small: 1 layer with 64 neurons
    ([128], "medium_1layer"),                 # 3. Medium: 1 layer with 128 neurons
    ([64, 64], "small_2layer"),               # 4. Small: 2 layers with 64 neurons each
    ([128, 64], "medium_2layer"),             # 5. Medium: 2 layers, narrowing
    
    # Medium depth networks (3 layers)
    ([64, 64, 64], "small_3layer"),          # 6. Small uniform: 3 layers with 64 neurons
    ([128, 128, 64], "medium_3layer"),       # 7. Medium narrowing: 3 layers
    ([256, 128, 64], "large_3layer"),        # 8. Large narrowing: 3 layers
    
    # Deep networks (4-5 layers)
    ([128, 128, 128, 64], "medium_4layer"),  # 9. Medium deep: 4 layers
    ([256, 256, 128, 64], "large_4layer"),   # 10. Large deep: 4 layers
    ([512, 256, 128, 64], "xlarge_4layer"),  # 11. Extra large: 4 layers
    ([128, 128, 128, 128, 64], "deep_5layer"), # 12. Deep: 5 layers
    
    # Wide but shallow
    ([512, 256], "wide_shallow"),            # 13. Wide but only 2 layers
    
    # Narrow but deep
    ([32, 32, 32, 32], "narrow_deep"),       # 14. Narrow but 4 layers
]

# Learning rates to test
learning_rates = [0.1, 0.01, 0.001, 0.0001]

# Max iterations to test
max_iterations_list = [100, 500, 1000]

# Number of trajectories (fixed for now)
num_trajectories = [20, 50, 100]

# Generate all combinations
for num_t in num_trajectories
    for (act_fn, act_name) in zip(activation_functions, activation_names)
        for (arch, arch_name) in architectures
            for lr in learning_rates
                for max_iter in max_iterations_list
                    push!(configs, (
                        activation = act_fn,
                        activation_name = act_name,
                        architecture = arch,
                        architecture_name = arch_name,
                        learning_rate = lr,
                        max_iterations = max_iter,
                        num_trajectories = num_t
                    ))
                end
            end
        end
    end
end

println("Total configurations to test: ", length(configs))
println("This will test:")
println("  - $(length(activation_functions)) activation functions")
println("  - $(length(architectures)) network architectures") 
println("  - $(length(learning_rates)) learning rates")
println("  - $(length(max_iterations_list)) iteration counts")
println("  - $(length(num_trajectories)) data counts")


# =============================================================================
# SETUP DIRECTORIES
# =============================================================================
results_dir = joinpath(@__DIR__, "..", "hyperparameter_results")
plots_dir = joinpath(results_dir, "plots")
models_dir = joinpath(results_dir, "models")

# Create directories if they don't exist
mkpath(plots_dir)
mkpath(models_dir)

# =============================================================================
# PLOTTING FUNCTION
# =============================================================================
"""
    create_training_plots(t_ref, data_plot, pred_final, loss_history, config, config_id, plots_dir)

Create comprehensive plots for neural ODE training results including:
- Position and velocity comparisons for all 3 bodies
- Training loss history
"""
function create_training_plots(t_ref, data_plot, pred_final, loss_history, config, config_id, plots_dir)
    # Create plots for all bodies (positions and velocities)
    plots_array = []
    
    # Position plots (top row) - Bodies 1, 2, 3
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
    
    # Velocity plots (bottom row) - Bodies 1, 2, 3
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
                     plot_title="Neural ODE Results - Config $config_id")
    
    # Create filename
    filename = "config_$(config_id)_$(config.activation_name)_$(config.architecture_name)_lr$(config.learning_rate)_iter$(config.max_iterations)_data$(config.num_trajectories)"
    
    # Save combined plot
    savefig(p_combined, joinpath(plots_dir, "$filename.png"))
    
    # Create and save separate loss history plot
    p_loss = plot(1:length(loss_history), loss_history, 
                 label="Training Loss", linewidth=2, yscale=:log10,
                 title="Training Loss - Config $config_id",
                 xlabel="Iteration", ylabel="Loss (log scale)",
                 size=(800, 600))
    
    # Add final loss annotation
    annotate!(p_loss, length(loss_history), loss_history[end], 
             text("Final: $(round(loss_history[end], sigdigits=4))", :left, 8))
    
    savefig(p_loss, joinpath(plots_dir, "$(filename)_loss.png"))
    
    return filename
end

# =============================================================================
# TRAINING FUNCTION
# =============================================================================
function train_configuration(config, dataset, config_id)
    println("\n" * "="^60)
    println("Configuration $config_id/$(length(configs)):")
    println("  Activation: $(config.activation_name)")
    println("  Architecture: $(config.architecture_name) - $(config.architecture)")
    println("  Learning rate: $(config.learning_rate)")
    println("  Max iterations: $(config.max_iterations)")
    println("  Data points: $(config.num_trajectories)")
    println("="^60)
    
    # Use first trajectory as reference
    t_ref = Float32.(dataset[1].t)
    u0_ref = Float32.(dataset[1].u0)
    
    # Build the neural network
    rng = Random.default_rng()
    Random.seed!(rng, 42)  # For reproducibility
    
    # Create layers based on architecture
    layers = []
    prev_size = length(u0_ref)
    for layer_size in config.architecture
        push!(layers, Lux.Dense(prev_size, layer_size, config.activation))
        prev_size = layer_size
    end
    push!(layers, Lux.Dense(prev_size, length(u0_ref)))  # Output layer
    
    dudt_nn = Lux.Chain(layers...)
    
    # Initialize parameters
    p, st = Lux.setup(rng, dudt_nn)
    ps = ComponentArray(p)
    
    # Define Neural ODE
    prob_neuralode = NeuralODE(dudt_nn, (t_ref[1], t_ref[end]), Tsit5(); saveat = t_ref)
    
    # Define loss function
    function loss_neuralode(p)
        total_loss = 0.0
        for trajectory in dataset[1:config.num_trajectories]
            u0_traj = Float32.(trajectory.u0)
            data_traj = Float32.(trajectory.data)
            pred = Array(prob_neuralode(u0_traj, p, st)[1])
            total_loss += sum(abs2, data_traj .- pred)
        end
        return total_loss / config.num_trajectories
    end
    
    # Track loss history
    loss_history = Float64[]
    
    # Callback function
    callback = function (p, l)
        push!(loss_history, l)
        if length(loss_history) % 10 == 0
            println("  Iteration $(length(loss_history)): Loss = $l")
        end
        return false
    end
    
    # Train the model
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, ps)
    
    try
        result = Optimization.solve(optprob,
                                  OptimizationOptimisers.Adam(config.learning_rate);
                                  callback = callback,
                                  maxiters = config.max_iterations)
        
        ps_trained = result.u
        final_loss = loss_history[end]
        
        # Generate predictions for plotting
        pred_final = Array(prob_neuralode(u0_ref, ps_trained, st)[1])
        data_plot = Float32.(dataset[1].data)
        
        # Create plots
        filename = create_training_plots(t_ref, data_plot, pred_final, loss_history, 
                                       config, config_id, plots_dir)
        
        # Save trained model
        jldsave(joinpath(models_dir, "$filename.jld2"); 
                ps_trained, st, config, loss_history, final_loss)
        
        return (config_id = config_id,
                config = config,
                final_loss = final_loss,
                loss_history = loss_history,
                success = true)
        
    catch e
        println("  Training failed: $e")
        return (config_id = config_id,
                config = config,
                final_loss = Inf,
                loss_history = loss_history,
                success = false)
    end
end

# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================
results = []

for (i, config) in enumerate(configs)
    result = train_configuration(config, dataset, i)
    push!(results, result)
    
    # Save intermediate results
    jldsave(joinpath(results_dir, "results_checkpoint.jld2"); results)
end

# =============================================================================
# ANALYZE RESULTS
# =============================================================================
println("\n" * "="^60)
println("TRAINING COMPLETE - SUMMARY")
println("="^60)

# Find best configuration
successful_results = filter(r -> r.success && r.final_loss < Inf, results)
if isempty(successful_results)
    println("No successful configurations found!")
else
    best_result = argmin(r -> r.final_loss, successful_results)
    
    println("\nBest configuration:")
    println("  Config ID: $(best_result.config_id)")
    println("  Activation: $(best_result.config.activation_name)")
    println("  Architecture: $(best_result.config.architecture_name) - $(best_result.config.architecture)")
    println("  Learning rate: $(best_result.config.learning_rate)")
    println("  Max iterations: $(best_result.config.max_iterations)")
    println("  Data points: $(best_result.config.num_trajectories)")
    println("  Final loss: $(best_result.final_loss)")
    
    # Create summary DataFrame
    df = DataFrame(
        config_id = [r.config_id for r in results],
        activation = [r.config.activation_name for r in results],
        architecture = [r.config.architecture_name for r in results],
        learning_rate = [r.config.learning_rate for r in results],
        max_iterations = [r.config.max_iterations for r in results],
        num_trajectories = [r.config.num_trajectories for r in results],
        final_loss = [r.final_loss for r in results],
        success = [r.success for r in results]
    )
    
    # Save results table
    CSV.write(joinpath(results_dir, "results_summary.csv"), df)
    
    # Create comparison plot of top 10 configurations
    sorted_results = sort(successful_results, by = r -> r.final_loss)
    top_n = min(10, length(sorted_results))
    
    p_comparison = plot(title="Top $top_n Configurations - Loss Curves",
                       xlabel="Iteration", ylabel="Loss (log scale)",
                       yscale=:log10, size=(1000, 600))
    
    for i in 1:top_n
        r = sorted_results[i]
        label = "$(r.config.activation_name)-$(r.config.architecture_name)-lr$(r.config.learning_rate)"
        plot!(p_comparison, r.loss_history, label=label, linewidth=2)
    end
    
    savefig(p_comparison, joinpath(results_dir, "top_configurations_comparison.png"))
    
    # Save final results
    jldsave(joinpath(results_dir, "final_results.jld2"); results, df, best_result)
end

println("\nResults saved in: $results_dir")
println("Plots saved in: $plots_dir")
println("Models saved in: $models_dir")