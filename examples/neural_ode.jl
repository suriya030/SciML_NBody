## Introduction to Neural ODEs
## Best explanation article I've found so far: https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
    OptimizationOptimisers, Random, Plots

rng = Random.default_rng()
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)


### UNderlying function which will be used to generate the data!

#du/dt = [du1/dt; du2/dt] = [u1^3 u2^3] * [ -0.1 2; -2 -0.1]
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)

## Generating the ground truth data
ode_data = Array(solve(prob_trueode, Tsit5(); saveat = tsteps))

plot(ode_data')


### Define the Neural ODE
dudt2 = Lux.Chain(x -> x .^ 3, Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
p, st = Lux.setup(rng, dudt2)


prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

### Define loss function as the difference between actual ground truth data and Neural ODE prediction
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot = true)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, ode_data[1, :]; label = "data")
        scatter!(plt, tsteps, pred[1, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot = true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05); callback = callback,
    maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(; initial_stepnorm = 0.01);
    callback, allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot = true)