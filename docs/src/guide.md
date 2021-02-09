# [Guide](@id guide)

To use this package to stress test your own system, the user has to provide the following:
- Implementations of the `BlackBox` system interface functions (outlined [here](@ref blackbox_interface))
- Implementations of the `GrayBox` simulator/environment interface functions (outlined [here](@ref graybox_interface))

## Problem Setup
Once the `GrayBox` and `BlackBox` interfaces are defined (see [Example](@ref example) for a full example), the user has access to the solvers and simply needs to set up the AST problem.

First, set up your simulation structure, where `YourSimulation <: GrayBox.Simulation` (note, you'll need to replace `YourSimulation` with your own structure you've defined as part of the `GrayBox` interface):
```julia
sim::GrayBox.Simulator = YourSimulation()
```

Then, set up the adaptive stress testing (AST) Markov decision process (MDP) structure, given your `sim` object:
```julia
mdp::ASTMDP = ASTMDP{ASTSeedAction}(sim) # could use `ASTSeedAction` or `ASTSampleAction`
```

#### [AST Action Type](@id ast_action_type)
- **`ASTSeedAction`**
    - This action type samples seeds for a random number generator (RNG), which means the `GrayBox.transition!` function must sample from the environment themselves and apply the transition.
        - Useful when it's difficult to access the individual sampled environment outputs.
- **`ASTSampleAction`**
    - This action type samples directory from the `GrayBox.Environment` and will pass the sample(s) (as `GrayBox.EnvironmentSample`) to the `GrayBox.transition!` function to be directly applied
        - Useful when you have full access to the simulation environment and can apply each sample directly in the transition function.

## Solving the AST MDP

Now you can choose your solver (see [Solvers](@ref solvers)) and run `solve` given the AST `mdp` (Markov decision process) to produce an online `planner` (no search has been performed yet):
```julia
solver = MCTSPWSolver()
planner = solve(solver, mdp)
```

## Searching for Failures
Once the problem is set up, you can search for failures using `search!` given your `planner`. This will return the best action trace it found.
```julia
action_trace = search!(planner)
```

## Playback Failures
Given either the `action_trace` or the top `k` performing action traces, using [`get_top_path(mdp)`](@ref), you can playback the particular action:
```julia
final_state = playback(planner, action_trace)
```

## [Metrics and Visualizations](@id metrics_visualizations)
Afterwards, you can look at performance metrics and visualizations (see [Metrics](@ref metrics) and [Visualization](@ref visualizations)):
```julia
print_metrics(planner)
```

To plot, first install the `PyPlot` and `Seaborn` packages and load them. We use [`Requires.jl`](https://github.com/JuliaPackaging/Requires.jl) to handle these dependencies.
```julia
using PyPlot
using Seaborn
```

You can plot episodic metrics, including running miss distance mean, minimum miss distance, and cumulative failures all over episode (i.e. iteration):
```julia
episodic_figures(planner)
```

You can also plot the miss distance distribution and log-likelihood distribution:
```julia
distribution_figures(planner)
```
