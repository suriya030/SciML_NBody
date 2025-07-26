# plotting_utils.jl

"""
This module provides utility functions for plotting results and printing summaries.
"""
module PlottingUtils

export create_training_plots, print_config_summary, print_run_summary

using Plots

"""
    print_config_summary(config, config_id, total_configs)

Prints a summary of the current hyperparameter configuration being tested.
"""
function print_config_summary(config, config_id, total_configs)
    println("\n" * "="^60)
    println("Configuration $config_id/$total_configs:")
    println("  Activation: $(config.activation_name)")
    println("  Architecture: $(config.architecture_name) - $(config.architecture)")
    println("  Learning rate: $(config.learning_rate)")
    println("  Max iterations: $(config.max_iterations)")
    println("  Data points: $(config.num_trajectories)")
    println("="^60)
end

"""
    print_run_summary(configs)

Prints a summary of the hyperparameter search space.
"""
function print_run_summary(configs)
    # Extract unique values to count
    num_act = length(unique(c.activation_name for c in configs))
    num_arch = length(unique(c.architecture_name for c in configs))
    num_lr = length(unique(c.learning_rate for c in configs))
    num_iter = length(unique(c.max_iterations for c in configs))
    num_data = length(unique(c.num_trajectories for c in configs))

    println("Total configurations to test: ", length(configs))
    println("This will test:")
    println("  - $num_act activation functions")
    println("  - $num_arch network architectures")
    println("  - $num_lr learning rates")
    println("  - $num_iter iteration counts")
    println("  - $num_data data counts")
end

"""
    create_training_plots(t_ref, data_plot, pred_final, loss_history, config, config_id, plots_dir)

Create and save comprehensive plots for a single neural ODE training run.
"""
function create_training_plots(t_ref, data_plot, pred_final, loss_history, config, config_id, plots_dir)
    plots_array = []
    
    # Position plots
    for body in 1:3
        idx_start = 3*(body-1) + 1
        p = plot(t_ref, data_plot[idx_start, :], label="True x", c=:red, lw=2)
        plot!(p, t_ref, pred_final[idx_start, :], label="Pred x", c=:red, ls=:dash, lw=2)
        plot!(p, t_ref, data_plot[idx_start+1, :], label="True y", c=:green, lw=2)
        plot!(p, t_ref, pred_final[idx_start+1, :], label="Pred y", c=:green, ls=:dash, lw=2)
        plot!(p, t_ref, data_plot[idx_start+2, :], label="True z", c=:blue, lw=2)
        plot!(p, t_ref, pred_final[idx_start+2, :], label="Pred z", c=:blue, ls=:dash, lw=2)
        title!(p, "Body $body Positions"); xlabel!(p, "Time"); ylabel!(p, "Position")
        push!(plots_array, p)
    end
    
    # Velocity plots
    for body in 1:3
        idx_start = 9 + 3*(body-1) + 1
        p = plot(t_ref, data_plot[idx_start, :], label="True vx", c=:red, lw=2)
        plot!(p, t_ref, pred_final[idx_start, :], label="Pred vx", c=:red, ls=:dash, lw=2)
        plot!(p, t_ref, data_plot[idx_start+1, :], label="True vy", c=:green, lw=2)
        plot!(p, t_ref, pred_final[idx_start+1, :], label="Pred vy", c=:green, ls=:dash, lw=2)
        plot!(p, t_ref, data_plot[idx_start+2, :], label="True vz", c=:blue, lw=2)
        plot!(p, t_ref, pred_final[idx_start+2, :], label="Pred vz", c=:blue, ls=:dash, lw=2)
        title!(p, "Body $body Velocities"); xlabel!(p, "Time"); ylabel!(p, "Velocity")
        push!(plots_array, p)
    end
    
    p_combined = plot(plots_array..., layout=(2,3), size=(1200, 800),
                     plot_title="Neural ODE Results - Config $config_id")
    
    filename_base = "config_$(config_id)_$(config.activation_name)_$(config.architecture_name)_lr$(config.learning_rate)_iter$(config.max_iterations)_data$(config.num_trajectories)"
    savefig(p_combined, joinpath(plots_dir, "$filename_base.png"))
    
    # Loss history plot
    p_loss = plot(1:length(loss_history), loss_history, 
                 label="Training Loss", lw=2, yscale=:log10,
                 title="Training Loss - Config $config_id",
                 xlabel="Iteration", ylabel="Loss (log scale)",
                 size=(800, 600))
    annotate!(p_loss, length(loss_history), loss_history[end], 
             text("Final: $(round(loss_history[end], sigdigits=4))", :left, 8))
    savefig(p_loss, joinpath(plots_dir, "$(filename_base)_loss.png"))
    
    return filename_base
end

end # module PlottingUtils
