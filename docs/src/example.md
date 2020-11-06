# [Example Problem](@id example)
[![Example Notebook](https://img.shields.io/badge/example-notebook-blue)](https://nbviewer.jupyter.org/github/sisl/POMDPStressTesting.jl/blob/master/notebooks/Walk1D.ipynb)

This section walks through a full example to illustrate how to use the POMDPStressTesting package. See the [Jupyter notebook](https://nbviewer.jupyter.org/github/sisl/POMDPStressTesting.jl/blob/master/notebooks/Walk1D.ipynb) for the full implementation.

Some definitions to note for this example problem:
- **System**: a one-dimensional walking agent
- **Environment**: distribution of random walking actions, sampled from a standard normal distribution $$\mathcal N(0.1)$$
- **Failure event**: agent walks outside of the ±10 region
- **Distance metric**: how close to the ±10 edge is the agent?

#### Problem Description (Walk1D)
This problem, called Walk1D, samples random walking distances from a standard normal distribution $$\mathcal N(0,1)$$ and defines failures as walking past a certain threshold (which is set to ±10 in this example). AST will either select the seed which deterministically controls the sampled value from the distribution (i.e. from the transition model) or will directly sample the provided environmental distributions. These action modes are determined by the seed-action or sample-action options (i.e. [`ASTSeedAction`](@ref ast_action_type) or [`ASTSampleAction`](@ref ast_action_type), respectively) . AST will guide the simulation to failure events using a notion of distance to failure, while simultaneously trying to find the set of actions that maximizes the log-likelihood of the samples.


> Refer to the [notebook](https://nbviewer.jupyter.org/github/sisl/POMDPStressTesting.jl/blob/master/notebooks/Walk1D.ipynb) for the full implementation.

> For the non-notebook version, see the Julia file [test/Walk1D.jl](https://github.com/mossr/POMDPStressTesting.jl/blob/master/test/Walk1D.jl)