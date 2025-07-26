# training_function.jl

"""
This module contains the core function for training the Neural ODE model
for a single hyperparameter configuration.
"""
module Training

export train_configuration

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization,
      OptimizationOptimisers, Random, JLD2

# Since this module will be called from the main script, we assume
# the plotting utils are available in the scope.
using ..PlottingUtils

"""
    train_configuration(config, dataset, config_id, total_configs, plots_dir, models_dir)

Trains a Neural ODE model for a single hyperparameter configuration,
saves the model and plots, and returns the results.
"""
function train_configuration(config, dataset, config_id, total_configs, plots_dir, models_dir)
    print_config_summary(config, config_id, total_configs)
    
    t_ref = Float32.(dataset[1].t)
    u0_ref = Float32.(dataset[1].u0)
    
    rng = Random.default_rng()
    Random.seed!(rng, 42)
    
    layers = []
    prev_size = length(u0_ref)
    for layer_size in config.architecture
        push!(layers, Lux.Dense(prev_size, layer_size, config.activation))
        prev_size = layer_size
    end
    push!(layers, Lux.Dense(prev_size, length(u0_ref)))
    
    dudt_nn = Lux.Chain(layers...)
    p, st = Lux.setup(rng, dudt_nn)
    ps = ComponentArray(p)
    
    prob_neuralode = NeuralODE(dudt_nn, (t_ref[1], t_ref[end]), Tsit5(); saveat = t_ref)
    
    function loss_neuralode(p)
        total_loss = 0.0f0
        for i in 1:config.num_trajectories
            u0_traj = Float32.(dataset[i].u0)
            data_traj = Float32.(dataset[i].data)
            pred = Array(prob_neuralode(u0_traj, p, st)[1])
            total_loss += sum(abs2, data_traj .- pred)
        end
        return total_loss / config.num_trajectories
    end
    
    loss_history = Float64[]
    
    callback = function (p, l)
        push!(loss_history, l)
        if length(loss_history) % 10 == 0
            println("  Iteration $(length(loss_history)): Loss = $l")
        end
        return false
    end
    
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
        
        pred_final = Array(prob_neuralode(u0_ref, ps_trained, st)[1])
        data_plot = Float32.(dataset[1].data)
        
        filename = create_training_plots(t_ref, data_plot, pred_final, loss_history, 
                                       config, config_id, plots_dir)
        
        jldsave(joinpath(models_dir, "$filename.jld2"); 
                ps_trained, st, config, loss_history, final_loss)
        
        return (config_id=config_id, config=config, final_loss=final_loss, 
                loss_history=loss_history, success=true)
        
    catch e
        println("  Training failed: $e")
        return (config_id=config_id, config=config, final_loss=Inf, 
                loss_history=loss_history, success=false)
    end
end

end # module Training
