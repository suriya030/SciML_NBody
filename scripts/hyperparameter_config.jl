# hyperparameter_config.jl

"""
This module defines the hyperparameter configurations for the training process.
"""
module HyperparameterConfig

export get_hyperparameter_configs

# Fix: Import Lux to get access to activation functions like tanh, sigmoid, relu
using Lux

"""
    get_hyperparameter_configs()

Generates and returns a list of all hyperparameter combinations to test.
Each configuration is a NamedTuple containing settings for activation,
architecture, learning rate, and more.
"""
function get_hyperparameter_configs()
    # Define all hyperparameter combinations to test
    configs = []

    # Activation functions to test (now accessible from Lux)
    activation_functions = [tanh, sigmoid, relu]
    activation_names = ["tanh", "sigmoid", "relu"]

    # Network architectures (input_size -> hidden_layers -> output_size)
    architectures = [
        ([32], "tiny_1layer"),
        ([64], "small_1layer"),
        ([128], "medium_1layer"),
        ([64, 64], "small_2layer"),
        ([128, 64], "medium_2layer"),
        ([64, 64, 64], "small_3layer"),
        ([128, 128, 64], "medium_3layer"),
        ([256, 128, 64], "large_3layer"),
        ([128, 128, 128, 64], "medium_4layer"),
        ([256, 256, 128, 64], "large_4layer"),
        ([512, 256, 128, 64], "xlarge_4layer"),
        ([128, 128, 128, 128, 64], "deep_5layer"),
        ([512, 256], "wide_shallow"),
        ([32, 32, 32, 32], "narrow_deep"),
    ]

    # Learning rates to test
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    # Max iterations to test
    max_iterations_list = [100, 500, 1000]

    # Number of trajectories
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
    
    return configs
end

end # module HyperparameterConfig
