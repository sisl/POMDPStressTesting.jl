# POMDPStressTesting.jl
Adaptive Stress Testing for [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)

# BlackBox Interface
To stress test a new system, the user has to define the `BlackBox` interface outlined in [`src/BlackBox.jl`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/src/BlackBox.jl)

The interface includes:
* `Simulation` type to hold simulation variables
* `initialize!(sim::Simulation)` to initialize/reset the system under test
* `evaluate!(sim::Simulation)` to evaluate/execute the system under test
* `transition_model!(sim::Simulation)` to transition the simulator, returning the log-likelihood
* `isevent!(sim::Simulation)` to indicate if a failure event occurred
* `miss_distance!(sim::Simulation)` to return how close we are to an event
* `isterminal!(sim::Simulation)` to indicate the simulation is in a terminal state

All of these functions can modify the `Simulation` object in place.


# Example

See example implementation of the AST interface for the Walk1D problem: [`test/Walk1D.jl`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/test/Walk1D.jl).

With an accompanying notebook: [`Walk1D.ipynb`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/notebooks/Walk1D.ipynb)

