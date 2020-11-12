# POMDPStressTesting
[![Build Status](https://travis-ci.org/sisl/POMDPStressTesting.jl.svg?branch=master)](https://travis-ci.org/sisl/POMDPStressTesting.jl) [![codecov](https://codecov.io/gh/sisl/POMDPStressTesting.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/POMDPStressTesting.jl)

This package is used to find likely failures in a black-box software system.
The package is integrated with the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) ecosystem; giving access to solvers, policies, and visualizations (although no prior knowledge of the POMDPs.jl package is needed to use POMDPStressTesting.jl—see the [Guide](@ref guide)). It uses a technique called adaptive stress testing (AST)[^1] to find likely failures using a distance metric to a failure event and the likelihood of an environment sample to guide the search.

A POMDP is a [partially observable Markov decision process](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process), which is a framework to define a sequential decision making problem where the true state is unobservable. In the context of this package, we use the POMDP acronym mainly to tie the package to the POMDPs.jl package, but the system that is stress tested can also be defined as a POMDP.

This package is intended to help developers stress test their systems before deployment into the real-world (see existing use cases for aircraft collision avoidance systems[^1] and aircraft trajectory prediction systems[^2]). It is also used for research purposes to expand on the AST concept by allowing additional solution methods to be explored and tested.

---

[^1] Ritchie Lee et al., *"Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning
"*, 2020. [https://arxiv.org/abs/1811.02188](https://arxiv.org/abs/1811.02188)

[^2] Robert J. Moss, Ritchie Lee, Nicholas Visser, Joachim Hochwarth, James G. Lopez, Mykel J. Kochenderfer, *"Adaptive Stress Testing of Trajectory Predictions in Flight Management Systems"*, DASC 2020. [https://arxiv.org/abs/2011.02559](https://arxiv.org/abs/2011.02559)

---

## Package Features
- Search for failures in a black-box [system](@ref system)
- Define probability distributions of your [simulation environment](@ref sim_env)
- Find likely system failures using a variety of [solvers](@ref solvers)
- Calculate and visualize [failure metrics](@ref metrics_visualizations)
- Replay found failures


## Contents
```@contents
Pages = [
    "installation.md",
    "guide.md",
    "solvers.md",
    "example.md",
    "contrib.md"
]
```

## [`BlackBox` System Definition](@id system)
A black-box system could be an external software executable, code written in Julia, or code written in another language.
The system is generally a sequential decision making system than can be stepped foward in time.
It is termed "black-box" because all we need is to be able to initialize it (using `BlackBox.initialize!`), evaluate or step the system forward in time (using `BlackBox.evaluate!`), and parse the output of the system to determine the distance metric (using `BlackBox.distance`), the failure event indication (using `BlackBox.isevent`), and whether the system is in a terminal state (using `BlackBox.isterminal`).
- See the [`BlackBox` interface](@ref blackbox_interface) for implementation details

The `BlackBox` system interface includes:
- `BlackBox.initialize!(sim::Simulation)` to initialize/reset the system under test
- `BlackBox.evaluate!(sim::Simulation)` to evaluate/execute the system under test
- `BlackBox.distance(sim::Simulation)` to return how close we are to an event
- `BlackBox.isevent(sim::Simulation)` to indicate if a failure event occurred
- `BlackBox.isterminal(sim::Simulation)` to indicate the simulation is in a terminal state


## [`GrayBox` Simulator/Environment Definition](@id sim_env)
The gray-box simulator and environment define the parameters of your simulation and the probability distributions governing your simulation environment. It is termed "gray-box" because we need access to the probability distributions of the environment in order to get the log-likelihood of a sample used by the simulator (which is ulimately used by the black-box system).
- See the [`GrayBox` interface](@ref graybox_interface) for implementation details.

The `GrayBox` simulator and environment interface includes:
- `GrayBox.Simulation` type to hold simulation variables
- `GrayBox.environment(sim::Simulation)` to return the collection of environment distributions
- `GrayBox.transition!(sim::Simulation)` to transition the simulator, returning the log-likelihood

## Failure and Distance Definition
A *failure* event of the system under test is defined be the user. The user defines the function `BlackBox.isevent` to return an boolean indicating a failure or not given the current state of the simulation. An example failure used in the context of AST would be a collision when stress testing autonomous vehicles or aircraft collision avoidance systems.

The real-valued *distance* metric is used to indicate "how close are we to a failure?" and is defined by the user in the `BlackBox.distance` function. This metric is used to guide the search process towards failures by receiving a signal of the distance to a failure. An example distance metric for the autonomous vehicle problem would be the distance between the autonomous vehicle and a pedestrian, where if a failure is a collision with a pedestrian then we'd like to minimize this distance metric to find failures.

## Citation

If you use this package for research purposes, please cite the following:

[![status](https://joss.theoj.org/papers/04dc39ea89e90938727d789a2e402b0b/status.svg)](https://joss.theoj.org/papers/04dc39ea89e90938727d789a2e402b0b)