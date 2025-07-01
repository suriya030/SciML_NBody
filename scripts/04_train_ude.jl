# scripts/04_train_ude.jl

using ComponentArrays, Lux, OrdinaryDiffEq, Optimization, OptimizationOptimisers,
      Random, JLD2

# Load the training data using a robust path
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")
t_train, data_train, u0_train = dataset[1]

# 1. Define the small neural network for the force law using Lux
rng = Random.default_rng()
force_ann = Lux.Chain(Lux.Dense(3, 16, tanh), Lux.Dense(16, 3))

# 2. Initialize the parameters (p_ann) and state (st_ann) for the Lux model
p_ann, st_ann = Lux.setup(rng, force_ann)

# 3. The UDE combines known physics with the neural network
function ude_n_body_system!(du, u, p, t)
    # p is a ComponentArray with p.phys and p.nn
    N = 3 # Number of bodies
    G = p.phys[4] # Gravitational constant
    p_nn = p.nn # Extract neural network parameters

    r = @view u[1:3*N]
    v = @view u[3*N+1:end]

    du[1:3*N] .= v # Known physics: dr/dt = v
    du[3*N+1:end] .= 0.0 # Initialize accelerations

    for i in 1:N
        for j in (i+1):N
            ri = @view r[3*(i-1)+1 : 3*i]
            rj = @view r[3*(j-1)+1 : 3*j]
            r_ij = rj - ri

            # Apply the Lux model, passing input, parameters, and state
            # The state `st_ann` is captured by the closure
            nn_force, _ = force_ann(r_ij, p_nn, st_ann)
            force_ij = G * nn_force

            du[3*(i-1)+1 : 3*i] .+= p.phys[j] * force_ij
            du[3*(j-1)+1 : 3*j] .-= p.phys[i] * force_ij
        end
    end
end

# 4. Combine physical and NN parameters into a named ComponentArray
p_phys = [1.0, 1.0, 1.0, 1.0] # m1, m2, m3, G
p_initial = ComponentArray(phys = p_phys, nn = p_ann)
tspan = (t_train[1], t_train[end])

ude_prob = ODEProblem(ude_n_body_system!, u0_train, tspan, p_initial)

# 5. Define the loss function
function loss_ude(p)
    # Use remake to solve the problem with new parameters `p`
    _prob = remake(ude_prob, p = p)
    sol = solve(_prob, Tsit5(), saveat = t_train)

    # Return high loss if solver failed
    if !successful_retcode(sol.retcode)
        return Inf, []
    end
    
    pred = Array(sol)
    loss = sum(abs2, data_train .- pred)
    return loss, pred # Return loss and prediction
end

# 6. Set up the callback function to monitor training
callback = function (p, l, pred)
    println("UDE Loss: ", l)
    # Return false to continue training
    return false
end

# 7. Use Optimization.jl to set up and run the training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_ude(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_initial)

println("Starting training for the UDE...")
result_ude = Optimization.solve(optprob,
                                OptimizationOptimisers.Adam(0.01);
                                callback = callback,
                                maxiters = 100)
println("Training complete.")