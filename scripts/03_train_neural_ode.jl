# scripts/03_train_neural_ode.jl

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization,
      OptimizationOptimisers, Random, JLD2

# Load the training data using a robust path
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")
t_train, data_train, u0_train = dataset[1]

# 1. Define the neural network using Lux.jl
rng = Random.default_rng()
dudt_nn = Lux.Chain(Lux.Dense(length(u0_train), 64, tanh),
                    Lux.Dense(64, 64, tanh),
                    Lux.Dense(64, length(u0_train)))

# 2. Initialize the parameters (ps) and state (st) for the Lux model
p, st = Lux.setup(rng, dudt_nn)
ps = ComponentArray(p) # Wrap parameters in a ComponentArray

# 3. Define the Neural ODE
# The state `st` is captured by the closure in the predict function
prob_neuralode = NeuralODE(dudt_nn, (t_train[1], t_train[end]), Tsit5();
                           saveat = t_train)

# 4. Define the prediction function, mirroring your example
function predict_neuralode(p)
    # Pass initial condition, parameters, and state to the NeuralODE problem
    Array(prob_neuralode(u0_train, p, st)[1])
end

# 5. Define the loss function
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data_train .- pred)
    # Return loss and prediction, as required by Optimization.jl's callback
    return loss, pred
end

# 6. Set up the callback function to monitor training
callback = function (p, l, pred)
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