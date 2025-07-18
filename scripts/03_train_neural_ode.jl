# scripts/03_train_neural_ode.jl

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization,
      OptimizationOptimisers, Random, JLD2

# Load the training data using a robust path
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")
t_train, data_train, u0_train = dataset[1]

# Convert data to Float32 for consistency with neural network
data_train = Float32.(data_train)
u0_train = Float32.(u0_train)
t_train = Float32.(t_train)

# 1. Define the neural network using Lux.jl
rng = Random.default_rng()
dudt_nn = Lux.Chain(Lux.Dense(length(u0_train), 64, tanh),
                    Lux.Dense(64, 64, tanh),
                    Lux.Dense(64, length(u0_train)))

# 2. Initialize the parameters (ps) and state (st) for the Lux model
p, st = Lux.setup(rng, dudt_nn)
ps = ComponentArray(p) # Wrap parameters in a ComponentArray

# 3. Define the Neural ODE
prob_neuralode = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5();
                           saveat = t_train)

# 4. Define the prediction function
function predict_neuralode(p)
    # Pass initial condition, parameters, and state to the NeuralODE problem
    Array(prob_neuralode(u0_train, p, st)[1])
end

# 5. Define the loss function - MUST return only scalar loss
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data_train .- pred)
    return loss  # Only return the scalar loss
end

# 6. Set up the callback function to monitor training
# We'll compute predictions inside the callback for monitoring
callback = function (p, l)
    println("Loss: ", l)
    # Return false to continue training
    return false
end

# 7. Use Optimization.jl to set up and run the training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ps)

println("Starting training for the Neural ODE...")
result_neuralode = Optimization.solve(optprob,
                                        OptimizationOptimisers.Adam(0.01);
                                        callback = callback,
                                        maxiters = 100)
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

# Plot first few dimensions to visualize fit
p1 = plot(t_train, data_train[1, :], label="True - Body 1 x", linewidth=2)
plot!(p1, t_train, pred_final[1, :], label="Predicted - Body 1 x", linestyle=:dash, linewidth=2)
plot!(p1, t_train, data_train[2, :], label="True - Body 1 y", linewidth=2)
plot!(p1, t_train, pred_final[2, :], label="Predicted - Body 1 y", linestyle=:dash, linewidth=2)
title!(p1, "Neural ODE Training Results")
xlabel!(p1, "Time")
ylabel!(p1, "Position")

# Save the plot
savefig(p1, joinpath(@__DIR__, "..", "plots", "neural_ode_training_results.png"))
display(p1)