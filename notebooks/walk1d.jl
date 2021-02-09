### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 92ce9460-f62b-11ea-1a8c-179776b5a0b4
using Distributions, Parameters, POMDPStressTesting, Latexify, PlutoUI

# ╔═╡ 2978b840-f62d-11ea-2ea0-19d7857208b1
md"""
# Black-Box Stress Testing
"""

# ╔═╡ 1e00230e-f630-11ea-3e40-bf8c852f78b8
# begin
#   using Pkg
#   pkg"registry add https://github.com/JuliaPOMDP/Registry"
#   pkg"add https://github.com/sisl/RLInterface.jl"
#   pkg"add https://github.com/sisl/POMDPStressTesting.jl"
# end
md"*Unhide for installation (waiting on Julia registry).*"

# ╔═╡ 40d3b1e0-f630-11ea-2160-01338d9f2209
md"""
To find failures in a black-box autonomous system, we can use the `POMDPStressTesting` package which is part of the POMDPs.jl ecosystem.

Various solvers—which adhere to the POMDPs.jl interface—can be used:
- `MCTSPWSolver` (MCTS with action progressive widening)
- `TRPOSolver` and `PPOSolver` (deep reinforcement learning policy optimization)
- `CEMSolver` (cross-entropy method)
- `RandomSearchSolver`
"""

# ╔═╡ 86f13f60-f62d-11ea-3241-f3f1ffe37d7a
md"""
## Simple Problem: One-Dimensional Walk
We define a simple problem for adaptive stress testing (AST) to find failures. This problem, called Walk1D, samples random walking distances from a standard normal distribution $\mathcal{N}(0,1)$ and defines failures as walking past a certain threshold (which is set to $\pm 10$ in this example). AST will either select the seed which deterministically controls the sampled value from the distribution (i.e. from the transition model) or will directly sample the provided environmental distributions. These action modes are determined by the seed-action or sample-action options. AST will guide the simulation to failure events using a notion of distance to failure, while simultaneously trying to find the set of actions that maximizes the log-likelihood of the samples.
"""

# ╔═╡ d3411dd0-f62e-11ea-27d7-1b2ed8edc415
md"""
## Gray-Box Simulator and Environment
The simulator and environment are treated as gray-box because we need access to the state-transition distributions and their associated likelihoods.
"""

# ╔═╡ e37d7542-f62e-11ea-0b61-513a4b44fc3c
md"""
##### Parameters
First, we define the parameters of our simulation.
"""

# ╔═╡ fd7fc880-f62e-11ea-15ac-f5407aeff2a6
@with_kw mutable struct Walk1DParams
    startx::Float64 = 0   # Starting x-position
    threshx::Float64 = 10 # +- boundary threshold
    endtime::Int64 = 30   # Simulate end time
end;

# ╔═╡ 012c2eb0-f62f-11ea-1637-c113ad01b144
md"""
##### Simulation
Next, we define a `GrayBox.Simulation` structure.
"""

# ╔═╡ 0d7049de-f62f-11ea-3552-214fc4e7ec98
@with_kw mutable struct Walk1DSim <: GrayBox.Simulation
    params::Walk1DParams = Walk1DParams() # Parameters
    x::Float64 = 0 # Current x-position
    t::Int64 = 0 # Current time ±
    distribution::Distribution = Normal(0, 1) # Transition distribution
end;

# ╔═╡ 11e445d0-f62f-11ea-305c-495272981112
md"""
### GrayBox.environment
Then, we define our `GrayBox.Environment` distributions. When using the `ASTSampleAction`, as opposed to `ASTSeedAction`, we need to provide access to the sampleable environment.
"""

# ╔═╡ 43c8cb70-f62f-11ea-1b0d-bb04a4176730
GrayBox.environment(sim::Walk1DSim) = GrayBox.Environment(:x => sim.distribution)

# ╔═╡ 48a5e970-f62f-11ea-111d-35694f3994b4
md"""
### GrayBox.transition!
We override the `transition!` function from the `GrayBox` interface, which takes an environment sample as input. We apply the sample in our simulator, and return the log-likelihood.
"""

# ╔═╡ 5d0313c0-f62f-11ea-3d33-9ded1fb804e7
function GrayBox.transition!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    sim.t += 1 # Keep track of time
    sim.x += sample[:x].value # Move agent using sampled value from input
    return logpdf(sample)::Real # Summation handled by `logpdf()`
end

# ╔═╡ 6e111310-f62f-11ea-33cf-b5e943b2f088
md"""
## Black-Box System
The system under test, in this case a simple single-dimensional moving agent, is always treated as black-box. The following interface functions are overridden to minimally interact with the system, and use outputs from the system to determine failure event indications and distance metrics.
"""

# ╔═╡ 7c84df7e-f62f-11ea-3b5f-8b090654df19
md"""
### BlackBox.initialize!
Now we override the `BlackBox` interface, starting with the function that initializes the simulation object. Interface functions ending in `!` may modify the `sim` object in place.
"""

# ╔═╡ 9b736bf2-f62f-11ea-0330-69ffafe9f200
function BlackBox.initialize!(sim::Walk1DSim)
    sim.t = 0
    sim.x = sim.params.startx
end

# ╔═╡ a380e250-f62f-11ea-363d-2bf2b59d5eed
md"""
### BlackBox.distance
We define how close we are to a failure event using a non-negative distance metric.
"""

# ╔═╡ be39db60-f62f-11ea-3a5c-bd57114455ff
BlackBox.distance(sim::Walk1DSim) = max(sim.params.threshx - abs(sim.x), 0)

# ╔═╡ bf8917b0-f62f-11ea-0e77-b58065b0da3e
md"""
### BlackBox.isevent
We define an indication that a failure event occurred.
"""

# ╔═╡ c5f03110-f62f-11ea-1119-81f5c9ec9283
BlackBox.isevent(sim::Walk1DSim) = abs(sim.x) ≥ sim.params.threshx

# ╔═╡ c378ef80-f62f-11ea-176d-e96e1be7736e
md"""
### BlackBox.isterminal
Similarly, we define an indication that the simulation is in a terminal state.
"""

# ╔═╡ cb5f7cf0-f62f-11ea-34ca-5f0656eddcd4
function BlackBox.isterminal(sim::Walk1DSim)
    return BlackBox.isevent(sim) || sim.t ≥ sim.params.endtime
end

# ╔═╡ e2f34130-f62f-11ea-220b-c7fc7de2c7e7
md"""
### BlackBox.evaluate!
Lastly, we use our defined interface to evaluate the system under test. Using the input sample, we return the log-likelihood, distance to an event, and event indication.
"""

# ╔═╡ f6213a50-f62f-11ea-07c7-2dcc383c8042
function BlackBox.evaluate!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation
    d::Real       = BlackBox.distance(sim) # Calculate miss distance
    event::Bool   = BlackBox.isevent(sim) # Check event indication
    return (logprob::Real, d::Real, event::Bool)
end

# ╔═╡ 01da7aa0-f630-11ea-1262-f50453455766
md"""
## AST Setup and Running
Setting up our simulation, we instantiate our simulation object and pass that to the Markov decision proccess (MDP) object of the adaptive stress testing formulation. We use Monte Carlo tree search (MCTS) with progressive widening on the action space as our solver. Hyperparameters are passed to `MCTSPWSolver`, which is a simple wrapper around the POMDPs.jl implementation of MCTS. Lastly, we solve the MDP to produce a planner. Note we are using the `ASTSampleAction`.
"""

# ╔═╡ fdf55130-f62f-11ea-33a4-a783b4d216dc
function setup_ast(seed=0)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = Walk1DSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10   # record top k best trajectories
    mdp.params.seed = seed  # set RNG seed for determinism

    # Hyperparameters for MCTS-PW as the solver
    solver = MCTSPWSolver(n_iterations=1000,        # number of algorithm iterations
                          exploration_constant=1.0, # UCT exploration
                          k_action=1.0,             # action widening
                          alpha_action=0.5,         # action widening
                          depth=sim.params.endtime) # tree depth

    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end;

# ╔═╡ 09c928f0-f631-11ea-3ef7-512a6bececcc
md"""
### Searching for Failures
After setup, we search for failures using the planner and output the best action trace.
"""

# ╔═╡ 17d0ed20-f631-11ea-2e28-3bb9ca9a445f
planner = setup_ast();

# ╔═╡ 1c47f652-f631-11ea-15f6-1b9b59700f36
action_trace = search!(planner)

# ╔═╡ 21530220-f631-11ea-3994-319c862d51f9
md"""
### Playback
We can also playback specific trajectories and print intermediate $x$-values.
"""

# ╔═╡ 3b282ae0-f631-11ea-309d-639bf4411bb3
playback_trace = playback(planner, action_trace, sim->sim.x, return_trace=true)

# ╔═╡ 7473adb0-f631-11ea-1c87-0f76b18a9ab6
failure_rate = print_metrics(planner)

# ╔═╡ b6244db0-f63a-11ea-3b48-89d427664f5e
md"""
## Other Solvers: Cross-Entropy Method
We can easily take our `ASTMDP` object (`planner.mdp`) and re-solve the MDP using a different solver—in this case the `CEMSolver`.
"""

# ╔═╡ e15cc1b0-f63a-11ea-2401-5321d48118c3
mdp = planner.mdp; # reused from above

# ╔═╡ f5a15af0-f63a-11ea-1dd7-593d7cb01ee4
cem_solver = CEMSolver(n_iterations=1000, episode_length=mdp.sim.params.endtime)

# ╔═╡ fb3fa610-f63a-11ea-2663-17224dc8aade
cem_planner = solve(cem_solver, mdp);

# ╔═╡ 09c9e0b0-f63b-11ea-2d50-4154e3432fa0
cem_action_trace = search!(cem_planner);

# ╔═╡ 46b40e10-f63b-11ea-2375-1976bb637d51
md"Notice the failure rate is about 10x of `MCTSPWSolver`."

# ╔═╡ 32fd5cf2-f63b-11ea-263a-39f013ef6d68
cem_failure_rate = print_metrics(cem_planner)

# ╔═╡ 801f8080-f631-11ea-0728-f15dddc3ef5d
md"""
## AST Reward Function
The AST reward function gives a reward of $0$ if an event is found, a reward of negative distance if no event is found at termination, and the log-likelihood during the simulation.
"""

# ╔═╡ 8f06f650-f631-11ea-1c52-697060322173
@latexify function R(p,e,d,τ)
    if τ && e
        return 0
    elseif τ && !e
        return -d
    else
        return log(p)
    end
end

# ╔═╡ 9463f6e2-f62a-11ea-1cef-c3fa7d4f19ad
md"""
## References
1. Robert J. Moss, Ritchie Lee, Nicholas Visser, Joachim Hochwarth, James G. Lopez, and Mykel J. Kochenderfer, "Adaptive Stress Testing of Trajectory Predictions in Flight Management Systems", *Digital Avionics Systems Conference, 2020.*
"""

# ╔═╡ 05be9a80-0877-11eb-3d88-efbd306754a2
PlutoUI.TableOfContents("POMDPStressTesting.jl")

# ╔═╡ Cell order:
# ╟─2978b840-f62d-11ea-2ea0-19d7857208b1
# ╟─1e00230e-f630-11ea-3e40-bf8c852f78b8
# ╠═92ce9460-f62b-11ea-1a8c-179776b5a0b4
# ╟─40d3b1e0-f630-11ea-2160-01338d9f2209
# ╟─86f13f60-f62d-11ea-3241-f3f1ffe37d7a
# ╟─d3411dd0-f62e-11ea-27d7-1b2ed8edc415
# ╟─e37d7542-f62e-11ea-0b61-513a4b44fc3c
# ╠═fd7fc880-f62e-11ea-15ac-f5407aeff2a6
# ╟─012c2eb0-f62f-11ea-1637-c113ad01b144
# ╠═0d7049de-f62f-11ea-3552-214fc4e7ec98
# ╟─11e445d0-f62f-11ea-305c-495272981112
# ╠═43c8cb70-f62f-11ea-1b0d-bb04a4176730
# ╟─48a5e970-f62f-11ea-111d-35694f3994b4
# ╠═5d0313c0-f62f-11ea-3d33-9ded1fb804e7
# ╟─6e111310-f62f-11ea-33cf-b5e943b2f088
# ╟─7c84df7e-f62f-11ea-3b5f-8b090654df19
# ╠═9b736bf2-f62f-11ea-0330-69ffafe9f200
# ╟─a380e250-f62f-11ea-363d-2bf2b59d5eed
# ╠═be39db60-f62f-11ea-3a5c-bd57114455ff
# ╟─bf8917b0-f62f-11ea-0e77-b58065b0da3e
# ╠═c5f03110-f62f-11ea-1119-81f5c9ec9283
# ╟─c378ef80-f62f-11ea-176d-e96e1be7736e
# ╠═cb5f7cf0-f62f-11ea-34ca-5f0656eddcd4
# ╟─e2f34130-f62f-11ea-220b-c7fc7de2c7e7
# ╠═f6213a50-f62f-11ea-07c7-2dcc383c8042
# ╟─01da7aa0-f630-11ea-1262-f50453455766
# ╠═fdf55130-f62f-11ea-33a4-a783b4d216dc
# ╟─09c928f0-f631-11ea-3ef7-512a6bececcc
# ╠═17d0ed20-f631-11ea-2e28-3bb9ca9a445f
# ╟─1c47f652-f631-11ea-15f6-1b9b59700f36
# ╟─21530220-f631-11ea-3994-319c862d51f9
# ╠═3b282ae0-f631-11ea-309d-639bf4411bb3
# ╠═7473adb0-f631-11ea-1c87-0f76b18a9ab6
# ╟─b6244db0-f63a-11ea-3b48-89d427664f5e
# ╠═e15cc1b0-f63a-11ea-2401-5321d48118c3
# ╠═f5a15af0-f63a-11ea-1dd7-593d7cb01ee4
# ╠═fb3fa610-f63a-11ea-2663-17224dc8aade
# ╠═09c9e0b0-f63b-11ea-2d50-4154e3432fa0
# ╟─46b40e10-f63b-11ea-2375-1976bb637d51
# ╠═32fd5cf2-f63b-11ea-263a-39f013ef6d68
# ╟─801f8080-f631-11ea-0728-f15dddc3ef5d
# ╠═8f06f650-f631-11ea-1c52-697060322173
# ╟─9463f6e2-f62a-11ea-1cef-c3fa7d4f19ad
# ╠═05be9a80-0877-11eb-3d88-efbd306754a2
