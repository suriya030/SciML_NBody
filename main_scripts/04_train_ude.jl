using ComponentArrays, Lux, DiffEqFlux, DifferentialEquations, Optimization,
      OptimizationOptimisers, OptimizationOptimJL, Random, JLD2, Plots

# Load dataset
println("Loading dataset...")
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")

# ==================== HYPERPARAMETERS ====================
# Data selection
sample_idx = 3

# Neural network architecture
hidden_layer_config = [64,64]  # Hidden layers for each interaction NN
# Initialization options:
# activation_function = tanh;       init_weight_func = Lux.glorot_uniform
# activation_function = sigmoid;    init_weight_func = Lux.glorot_uniform
# activation_function = relu;     init_weight_func = Lux.kaiming_uniform
activation_function = swish;    init_weight_func = Lux.glorot_uniform
init_bias_func = Lux.zeros32
# Training parameters
learning_rate_adam = 0.001     # Reduced from 0.001 for better stability
max_iterations_adam = 3500
learning_rate_bfgs = 0.001
max_iterations_bfgs = 100

# Callback settings
print_every = 10
plot_every = 500

# Random seed for reproducibility
random_seed = 1234
# ========================================================

# Set random seed
Random.seed!(random_seed)
rng = Random.default_rng()

# Select sample for training
t_ref = Float32.(dataset[sample_idx].t)
u0_ref = Float32.(dataset[sample_idx].u0)
data_ref = Float32.(dataset[sample_idx].data)

# Helper function to create neural networks with consistent architecture
function create_interaction_nn(input_dim, output_dim, hidden_layers, activation_fn;
                              init_weight, init_bias)
    layers = []
    current_dim = input_dim
    
    for hidden_dim in hidden_layers
        push!(layers, Lux.Dense(current_dim, hidden_dim, activation_fn;
                               init_weight=init_weight, init_bias=init_bias))
        current_dim = hidden_dim
    end
    
    push!(layers, Lux.Dense(current_dim, output_dim;
                           init_weight=init_weight, init_bias=init_bias))
    
    return Lux.Chain(layers...)
end

# Create neural networks for each pairwise interaction
# Each NN takes relative positions (3D) as input and outputs force components (3D)

# Neural Network for Body 1-2 interaction
NN_12 = create_interaction_nn(3, 3, hidden_layer_config, activation_function;
                             init_weight=init_weight_func, init_bias=init_bias_func)
p_12, st_12 = Lux.setup(rng, NN_12)

# Neural Network for Body 1-3 interaction
NN_13 = create_interaction_nn(3, 3, hidden_layer_config, activation_function;
                             init_weight=init_weight_func, init_bias=init_bias_func)
p_13, st_13 = Lux.setup(rng, NN_13)

# Neural Network for Body 2-3 interaction
NN_23 = create_interaction_nn(3, 3, hidden_layer_config, activation_function;
                             init_weight=init_weight_func, init_bias=init_bias_func)
p_23, st_23 = Lux.setup(rng, NN_23)

# Combine all parameters into a single ComponentArray
p_vec = ComponentArray(
    nn_12 = p_12,
    nn_13 = p_13,
    nn_23 = p_23
)

# Print model information
println("""
✅ UDE 3-Body Configuration:
---------------------------------
Sample Index:         $sample_idx
Time Span:           $(t_ref[1]) to $(t_ref[end]) s
Timesteps:           $(length(t_ref))

Neural Network Architecture:
- Input Dimension:    3 (relative positions)
- Hidden Layers:      $hidden_layer_config
- Output Dimension:   3 (force components)
- Activation:         $activation_function
- Weight Init:        $init_weight_func
- Bias Init:          $init_bias_func

Training Parameters:
- Adam LR:           $learning_rate_adam
- Adam Iterations:   $max_iterations_adam
- BFGS LR:           $learning_rate_bfgs
- BFGS Iterations:   $max_iterations_bfgs
---------------------------------
""")

# Define the UDE system
function n_body_ude!(du, u, p, t)
    # For 3 bodies
    N = 3
    
    # Extract positions and velocities
    r1 = u[1:3]
    r2 = u[4:6]
    r3 = u[7:9]
    v1 = u[10:12]
    v2 = u[13:15]
    v3 = u[16:18]
    
    # The derivative of position is velocity
    du[1:3] = v1
    du[4:6] = v2
    du[7:9] = v3
    
    # Initialize accelerations
    a1 = zeros(Float32, 3)
    a2 = zeros(Float32, 3)
    a3 = zeros(Float32, 3)
    
    # Body 1-2 interaction
    r_12 = r2 - r1
    force_12 = NN_12(r_12, p.nn_12, st_12)[1]
    a1 += force_12
    a2 -= force_12  # Newton's third law
    
    # Body 1-3 interaction
    r_13 = r3 - r1
    force_13 = NN_13(r_13, p.nn_13, st_13)[1]
    a1 += force_13
    a3 -= force_13  # Newton's third law
    
    # Body 2-3 interaction
    r_23 = r3 - r2
    force_23 = NN_23(r_23, p.nn_23, st_23)[1]
    a2 += force_23
    a3 -= force_23  # Newton's third law
    
    # Set accelerations
    du[10:12] = a1
    du[13:15] = a2
    du[16:18] = a3
end

# Create ODE problem
tspan = (t_ref[1], t_ref[end])
prob_ude = ODEProblem(n_body_ude!, u0_ref, tspan, p_vec)

# Helper function to extract parameters from optimization state
function get_params(θ)
    if θ isa ComponentArray
        return θ
    else
        # If it's an OptimizationState or similar, extract the parameters
        return θ.u
    end
end

# Prediction function
function predict_ude(θ)
    p = get_params(θ)
    sol = solve(prob_ude, Tsit5(), p=p, saveat=t_ref, 
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    return Array(sol)
end

# Loss function
function loss_ude(θ)
    pred = predict_ude(θ)
    loss = sum(abs2, data_ref .- pred)
    return loss
end

# Visualization function
function plot_ude_predictions(θ; iteration=nothing, save_path=nothing, display_plot=false)
    pred = predict_ude(θ)
    
    plots_array = []
    for body in 1:3
        # Position Plot
        p_pos = plot(title="Body $body Positions", legend=false)
        for (i, coord, color) in zip(1:3, ["x", "y", "z"], [:red, :green, :blue])
            idx = 3*(body-1) + i
            plot!(p_pos, t_ref, data_ref[idx, :], lw=2, c=color)
            plot!(p_pos, t_ref, pred[idx, :], ls=:dash, lw=2, c=color)
        end
        
        # Add legend only to the first body's position plot
        if body == 1
            plot!(p_pos, [], [], label="True", color=:black, lw=2, legend=:bottomright)
            plot!(p_pos, [], [], label="Predicted", color=:black, ls=:dash, lw=2)
        end
        
        push!(plots_array, p_pos)

        # Velocity Plot
        p_vel = plot(title="Body $body Velocities", legend=false)
        for (i, coord, color) in zip(1:3, ["vx", "vy", "vz"], [:red, :green, :blue])
            idx = 9 + 3*(body-1) + i
            plot!(p_vel, t_ref, data_ref[idx, :], lw=2, c=color)
            plot!(p_vel, t_ref, pred[idx, :], ls=:dash, lw=2, c=color)
        end
        push!(plots_array, p_vel)
    end

    title_str = isnothing(iteration) ? "UDE 3-Body Results" : "UDE 3-Body - Iteration $iteration"
    p_combined = plot(plots_array..., layout=(3,2), size=(800, 800),
                      plot_title=title_str)
    
    if !isnothing(save_path)
        savefig(p_combined, save_path)
    end
    
    if display_plot
        display(p_combined)
    end
    
    return p_combined
end

# Training callback
iter = 0
loss_history = Float32[]

function callback_ude(θ, l)
    global iter
    iter += 1
    push!(loss_history, l)
    
    if iter % print_every == 0
        println("Iteration $iter: Loss = $l")
    end
    
    if iter % plot_every == 0
        plot_ude_predictions(θ; iteration=iter, display_plot=true)
    end
    
    return false
end

# Calculate and display initial loss
initial_loss = loss_ude(p_vec)
println("\nInitial loss: $initial_loss")

# Plot initial predictions
println("Plotting initial predictions...")
plot_ude_predictions(p_vec; iteration=0, display_plot=true)

# Adam optimization
println("\nStarting Adam optimization...")
training_time_adam = @elapsed begin
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_ude(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_vec)
    
    result_adam = Optimization.solve(optprob, 
                                    OptimizationOptimisers.Adam(learning_rate_adam), 
                                    callback = callback_ude, 
                                    maxiters = max_iterations_adam)
end

println("\nAdam optimization complete.")
println("Training time: $(round(training_time_adam, digits=2)) seconds")
println("Final Adam loss: $(loss_history[end])")

# BFGS fine-tuning
println("\nStarting BFGS fine-tuning...")
iter = 0  # Reset iteration counter for BFGS
training_time_bfgs = @elapsed begin
    optprob2 = remake(optprob; u0 = result_adam.u)
    result_bfgs = Optimization.solve(optprob2, 
                                    OptimizationOptimJL.BFGS(initial_stepnorm=learning_rate_bfgs),
                                    callback = callback_ude,
                                    maxiters = max_iterations_bfgs)
end

println("\nBFGS optimization complete.")
println("Training time: $(round(training_time_bfgs, digits=2)) seconds")
println("Final loss: $(loss_history[end])")

# Save results
ps_trained = result_bfgs.u
model_path = joinpath(@__DIR__, "..", "data", "trained_ude_3body.jld2")
jldsave(model_path;
        ps_trained, st_12, st_13, st_23, NN_12, NN_13, NN_23,
        hyperparameters = Dict(
            "hidden_layers" => hidden_layer_config,
            "activation" => string(activation_function),
            "learning_rate_adam" => learning_rate_adam,
            "learning_rate_bfgs" => learning_rate_bfgs,
            "sample_idx" => sample_idx
        ),
        loss_history)
println("\nSaved trained UDE model to: $model_path")

# Create plots directory if it doesn't exist
plots_dir = joinpath(@__DIR__, "..", "plots")
!isdir(plots_dir) && mkdir(plots_dir)

# Final visualization
final_plot_path = joinpath(plots_dir, "ude_3body_final_results.png")
plot_ude_predictions(ps_trained; save_path=final_plot_path, display_plot=true)

# Plot loss history
loss_plot = plot(loss_history, 
                 xlabel="Iteration", 
                 ylabel="Loss", 
                 title="UDE Training Loss",
                 yscale=:log10,
                 lw=2,
                 label="Loss")
vline!([max_iterations_adam], ls=:dash, label="BFGS start", color=:red)
savefig(loss_plot, joinpath(plots_dir, "ude_loss_history.png"))
display(loss_plot)

# Compare with true gravitational force
function analyze_learned_forces(θ, G_true=1.0)
    p = get_params(θ)
    r_test = range(-2, 2, length=100)
    
    # Test along different axes
    fig = plot(layout=(1,3), size=(1200, 400))
    
    for (axis_idx, axis_name) in enumerate(["x", "y", "z"])
        forces_learned = Float32[]
        forces_true = Float32[]
        r_vals = Float32[]
        
        for r in r_test
            if abs(r) > 0.1  # Avoid singularity
                r_vec = zeros(Float32, 3)
                r_vec[axis_idx] = r
                
                # Learned force (using NN_12 as example)
                f_learned = NN_12(r_vec, p.nn_12, st_12)[1][axis_idx]
                push!(forces_learned, f_learned)
                
                # True gravitational force (assuming unit masses and G)
                f_true = G_true * r / abs(r)^3
                push!(forces_true, f_true)
                push!(r_vals, r)
            end
        end
        
        plot!(fig[axis_idx], r_vals, forces_learned, 
              label="Learned", lw=2, color=:blue)
        plot!(fig[axis_idx], r_vals, forces_true, 
              label="True Gravity", lw=2, ls=:dash, color=:red)
        xlabel!(fig[axis_idx], "Relative Position ($axis_name)")
        ylabel!(fig[axis_idx], "Force ($axis_name-component)")
        title!(fig[axis_idx], "Force along $axis_name-axis")
    end
    
    savefig(fig, joinpath(plots_dir, "ude_force_comparison.png"))
    display(fig)
end

println("\nAnalyzing learned forces...")
analyze_learned_forces(ps_trained)

println("\n✅ UDE training complete!")