# [Solvers](@id solvers)
This section describes the solution methods (i.e. solvers) used to in the AST problem formulation to search for likely failure events.

Several solvers are implemented:

## Reinforcement learning
Monte Carlo tree search (MCTS) is used as the search algorithm in the standard AST formulation.
We piggy-back on the [MCTS.jl](https://github.com/JuliaPOMDP/MCTS.jl) package with modification specific to AST.
Modifications to MCTS are described here: [^1]
* [`MCTSPWSolver`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/src/solvers/mcts.jl)

[^1] Robert J. Moss, Ritchie Lee, Nicholas Visser, Joachim Hochwarth, James G. Lopez, Mykel J. Kochenderfer, *"Adaptive Stress Testing of Trajectory Predictions in Flight Management Systems"*, DASC 2020. [https://arxiv.org/abs/2011.02559](https://arxiv.org/abs/2011.02559)

## Deep reinforcement learning
Deep reinforcement learning solvers include *trust region policy optimization* (TRPO) [^2] and *proximal policy optimization* (PPO) [^3].
We'd like to thank [Shreyas Kowshik's](https://github.com/shreyas-kowshik/RL-baselines.jl) for their initial Julia implementation of these methods.
* [`TRPOSolver`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/src/solvers/drl/trpo.jl)
* [`PPOSolver`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/src/solvers/drl/ppo.jl)

[^2] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel, *"Trust Region Policy Optimization"*, ICML 2015. [https://arxiv.org/abs/1502.05477](https://arxiv.org/abs/1502.05477)

[^3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov, *"Proximal Policy Optimization"*, 2017. [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

## Stochastic optimization
Solvers that use stochastic optimization include the [cross-entropy method](https://en.wikipedia.org/wiki/Cross-entropy_method) solver `CEMSolver`. This solution method is adapted from the [CrossEntropyMethod.jl](https://github.com/sisl/CrossEntropyMethod.jl) package by Anthony Corso.
* [`CEMSolver`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/src/solvers/cem.jl)


## Baselines
Baseline solvers are used for comparison to more sophisticated search methods. Currently, the only baseline solver is the `RandomSearchSolver` which uses Monte Carlo rollouts of a random policy to search for failures.
* [`RandomSearchSolver`](https://github.com/mossr/POMDPStressTesting.jl/blob/master/src/solvers/random_search.jl)


