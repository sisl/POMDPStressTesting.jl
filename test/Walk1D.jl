using Revise # DEBUG

# Based on Ritchie Lee's Walk1D in AdaptiveStressTesting.jl examples.

using POMDPStressTesting
using Distributions
using Random
using POMDPs
using MCTS

## AST formulation


## Black-box system under test


mutable struct Walk1DParams
	startx::Float64 # Starting x-position
	threshx::Float64 # +- boundary threshold
	endtime::Int64 # Simulate end time
	logging::Bool # State history logging indication

	Walk1DParams() = new(1, 10, 20, false)
end


mutable struct Walk1DSim <: BlackBox.Simulation
	p::Walk1DParams # Parameters
	x::Float64 # x-position
	t::Int64 # time
	distribution::Distribution
	history::Vector{Float64} # Log of history

	Walk1DSim(p::Walk1DParams, σ::Float64) = Walk1DSim(p, Normal(0.0, σ)) # Zero-mean Gaussian
	Walk1DSim(p::Walk1DParams, distribution::Distribution) = new(p, p.startx, 0, distribution, Float64[])
end


# Override from BlackBox
function BlackBox.initialize(sim::Walk1DSim)
	sim.t = 0
	sim.x = sim.p.startx
	empty!(sim.history)
	if sim.p.logging
		push!(sim.history, sim.x)
	end
end


# Override from BlackBox
function BlackBox.transition_model(sim::Walk1DSim)
	sample = rand(sim.distribution) # Sample value from distribution
	prob = pdf(sim.distribution, sample) # Get probability of sample
	return (prob, sample)
end


# Override from BlackBox
BlackBox.isevent(sim::Walk1DSim) = abs(sim.x) >= sim.p.threshx


# Override from BlackBox
BlackBox.miss_distance(sim::Walk1DSim) = max(sim.p.threshx - abs(sim.x), 0) # Non-negative


# Override from BlackBox
BlackBox.isterminal(sim::Walk1DSim) = BlackBox.isevent(sim) || sim.t >= sim.p.endtime


# Override from BlackBox
function BlackBox.evaluate(sim::Walk1DSim)
	sim.t += 1
	(prob::Float64, sample::Float64) = BlackBox.transition_model(sim)
	sim.x += sample
	miss_distance = BlackBox.miss_distance(sim)
	# @show sim.x, miss_distance, prob
	if sim.p.logging
		push!(sim.history, sim.x)
	end
	return (prob, BlackBox.isevent(sim), miss_distance)
end


function runtest()
	max_steps = 25 # Simulation end-time
	rsg_length = 2 # Number of unique available random seeds
	seed = 1 # RNG seed
	σ = 1.0 # Standard deviation

	# Setup black-box specific simulation parameters
	sim_params::Walk1DParams = Walk1DParams()
	sim_params.startx = 1.0
	sim_params.threshx = 10.0
	sim_params.endtime = max_steps
	sim_params.logging = true

	# Create black-box simulation object
	sim::BlackBox.Simulation = Walk1DSim(sim_params, σ)

	# AST specific parameters
	top_k::Int = 10 # Save top performing paths
	ast_params::AST.Params = AST.Params(max_steps, rsg_length, seed, top_k)

	# AST MDP formulation object
	mdp::AST.ASTMDP = AST.ASTMDP(ast_params, sim)
	# mdp.reset_rsg = AST.RandomSeedGenerator.RSG()

	# @requirements_info MCTSSolver() mdp

	rng = MersenneTwister(seed) # Unused. TODO local vs. global seed (i.e. use this)

	# MCTS with DPW solver parameters
	# TODO: AST version of this as a wrapper (i.e. sets required parameters)
	solver = MCTS.DPWSolver(
			estimate_value=AST.rollout, # TODO: required.
			depth=max_steps,
			enable_state_pw=false, # Custom fork of MCTS.jl (PR submitted) # TODO: best practice/required.
			exploration_constant=10.0,
			k_action=0.1,
			alpha_action=0.85,
			n_iterations=1000,
			# next_action=AST.next_action, # Unnecessary, implemented by MCTS.jl
			reset_callback=AST.go_to_state, # Custom fork of MCTS.jl # TODO: required.
			tree_in_info=true)#, rng=rng)

	planner = solve(solver, mdp)

	# s = initialstate(mdp, rng) # rng not used
	# a = action(planner, s)

	# Playback the best path in the tree
	# AST.playback(mdp) # TODO: export playback

	# display(sim.history)

	return (planner, mdp, sim, solver)
end

(planner, mdp, sim, solver) = runtest();