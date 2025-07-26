# hyperparameter_tuning.jl

# =============================================================================
# SETUP
# =============================================================================
# Include and use the modularized code
include("hyperparameter_config.jl")
include("plotting_utils.jl")
include("training_functions.jl")

using .HyperparameterConfig
using .PlottingUtils
using .Training

# Load necessary top-level libraries
using JLD2, DataFrames, CSV, Plots

println("Loading training data...")
dataset = load(joinpath(@__DIR__, "..", "data", "nbody_dataset.jld2"), "dataset")

# =============================================================================
# GET CONFIGURATIONS AND SETUP DIRECTORIES
# =============================================================================
configs = get_hyperparameter_configs()
print_run_summary(configs)

results_dir = joinpath(@__DIR__, "..", "hyperparameter_results")
plots_dir = joinpath(results_dir, "plots")
models_dir = joinpath(results_dir, "models")

mkpath(plots_dir)
mkpath(models_dir)

# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================
results = []
total_configs = length(configs)

for (i, config) in enumerate(configs)
    result = train_configuration(config, dataset, i, total_configs, plots_dir, models_dir)
    push!(results, result)
    
    # Save intermediate results after each run
    jldsave(joinpath(results_dir, "results_checkpoint.jld2"); results)
end

# =============================================================================
# END OF SCRIPT
# =============================================================================
println("\n" * "="^60)
println("TRAINING SWEEP COMPLETE")
println("="^60)
println("All configurations have been tested.")
println("Results, models, and plots are saved in: $results_dir")
println("You can now analyze the 'results_checkpoint.jld2' file manually.")

