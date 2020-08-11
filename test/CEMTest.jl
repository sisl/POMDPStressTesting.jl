using Revise # DEBUG

# Based on Ritchie Lee's Walk1D in AdaptiveStressTesting.jl examples.

using POMDPStressTesting
using Distributions
using Random
using POMDPs
using MCTS
using PyPlot

## AST formulation

include("C:\\Users\\mossr\\Documents\\Stanford\\3-Spring-2020\\AA222-CS361\\project\\code\\sierra.jl")

global LIMITS = (X=[-30,30], Y=[-30,30])

## Black-box system under test
# Algorithms for Optimization, Algorithm 8.7, Page 135
"""
    The cross-entropy method, which takes an objective
    function `f` to be minimized, a proposal distribution
    `P`, an iteration count `k_max`, a sample size `m`, and
    the number of samples to use when refitting the
    distribution `m_elite`. It returns the updated distribution
    over where the global minimum is likely to exist.
"""
function cross_entropy_method(f, P; m=10, m_elite=5, k_max=10, ϵ=1e-9, plot=true)
    bᵥ = Inf
    bₓ = missing

    μ = mean(P)
    for k in 1:k_max
        samples = rand(P, m)
        Y = [f(samples[:,i]) for i in 1:m]
        order = sortperm(Y)
        elite = samples[:,order[1:m_elite]]

        # Top elite.
        yₜ = Y[order[1]] # top elite y value
        if yₜ < bᵥ
            # Found better.
            bᵥ = yₜ
            bₓ = samples[:,order[1]]
        end

        # Plotting
        if plot
            plot_cem(P, samples, elite)
        end

        try
            P = fit(typeof(P), elite)
        catch err
            @warn(err); @show k
        end

#=
        if norm(μ - mean(P)) < ϵ
            println("Converged ($k)")
            break
        else
            # println(norm(μ - mean(P)))
        end
=#
        μ = mean(P)
    end

    # return (P, bₓ, bᵥ)
    return f(mean(P))
end



mutable struct CEMParams
	m::Int # number of evaluations per iteration
	m_elite::Int # number of elite samples to fit
	fthresh::Float64 # mean stopping threshold
	Σthresh::Float64 # covariance stopping threshold
	endtime::Int64 # simulate end time
	logging::Bool # state history logging indication

	# CEMParams() = new(5, 3, 1e-5, 1e-3, 10, false)
	CEMParams() = new(10, 3, 0.001, 1e-3, 10, false)
end


mutable struct CEMSim <: BlackBox.Simulation
	p::CEMParams # parameters
	f::Function # objective function
	μ::Vector{Float64} # mean
	t::Int64 # time
	distribution::Distribution
	samples::Matrix # samples
	elite::Matrix # elite samples
	history::Vector{Any} # log of history

	CEMSim(p::CEMParams) = CEMSim(p, MvNormal([0.,0], [200 0; 0. 200])) # Zero-mean multivariate Gaussian
	CEMSim(p::CEMParams, distribution::Distribution) = new(p, sierra, distribution.μ, 0, distribution, Matrix(undef,0,0), Matrix(undef,0,0), Any[])
end


# Override from BlackBox
function BlackBox.initialize!(sim::CEMSim)
	sim.t = 0
	sim.distribution = MvNormal([0.,0], [200 0; 0. 200]) # TODO. Shared.
	sim.μ = sim.distribution.μ
	sim.samples = Matrix(undef,0,0)
	sim.elite = Matrix(undef,0,0)
	empty!(sim.history)
	if sim.p.logging
		push!(sim.history, (μ = sim.μ, Σ = sim.distribution.Σ, elite = sim.elite, samples = sim.samples))
	end
end


# Override from BlackBox
function BlackBox.transition_model!(sim::CEMSim)
	
	# Cross-entropy method.
    samples = rand(sim.distribution, sim.p.m) # Draw samples from distribution
    Y = [sim.f(samples[:,i]) for i in 1:sim.p.m]
    elite = samples[:,sortperm(Y)[1:sim.p.m_elite]]
    sim.distribution = try fit(typeof(sim.distribution), elite) catch err; sim.distribution end

    sim.samples = samples
    sim.elite = elite

	logprob = try logpdf(sim.distribution, samples) catch err; 0 end # Get probability of samples

	return sum(logprob) # TODO. log?
end


# # Override from BlackBox
BlackBox.isevent!(sim::CEMSim) = all(abs(sim.f(sim.μ) - sim.f([0,0])) .<= [sim.p.fthresh, sim.p.fthresh]) && all(map(m->m <= sim.p.Σthresh, abs.(sim.distribution.Σ.mat)))
# abs(sim.μ) <= sim.p.μthresh && 

# # Override from BlackBox
BlackBox.miss_distance!(sim::CEMSim) = abs(sim.f(sim.μ)) # TODO. -sim.(f(sim.μ)) # negative instead of abs()
# max(sim.p.threshx - abs(sim.x), 0) # Non-negative

# Override from BlackBox
BlackBox.isterminal!(sim::CEMSim) = BlackBox.isevent!(sim) || sim.t >= sim.p.endtime


# Override from BlackBox
function BlackBox.evaluate!(sim::CEMSim)
	sim.t += 1
	logprob::Float64 = BlackBox.transition_model!(sim)
	# sim.x += sample
	sim.μ = sim.distribution.μ
	miss_distance = BlackBox.miss_distance!(sim)
	# @show sim.μ, miss_distance, logprob
	if sim.p.logging
		push!(sim.history, (μ = sim.μ, Σ = sim.distribution.Σ, elite = sim.elite, samples = sim.samples))
	end
	event::Bool = BlackBox.isevent!(sim)
	if event
		print("\rIsEvent!: $(sim.f(sim.μ) - sim.f([0,0]))")
	end
	return (logprob, event, miss_distance)
end


function runtest()
	max_steps = 25 # Simulation end-time
	rsg_length = 2 # Number of unique available random seeds
	seed = 1 # RNG seed

	# Setup black-box specific simulation parameters
	sim_params::CEMParams = CEMParams()
	sim_params.endtime = max_steps
	sim_params.logging = true

	# Create black-box simulation object
	sim::BlackBox.Simulation = CEMSim(sim_params)

	# AST specific parameters
	top_k::Int = 10 # Save top performing paths
	distance_reward::Bool = false
	debug::Bool = false
	ast_params::AST.Params = AST.Params(max_steps, rsg_length, seed, top_k, distance_reward, debug)

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
			# n_iterations=500,
			n_iterations=50,
			# next_action=AST.next_action, # Unnecessary, implemented by MCTS.jl
			reset_callback=AST.go_to_state, # Custom fork of MCTS.jl # TODO: required.
			tree_in_info=true)#, rng=rng)

	planner = solve(solver, mdp)

	# s = initialstate(mdp, rng) # rng not used
	# a = action(planner, s)

	# Playback the best path in the tree
	# AST.playback(mdp) # TODO: export playback

	# display(sim.history)

	# AST.plotout(mdp, planner)

	return (planner, mdp, sim, solver)
end

(planner, mdp, sim, solver) = runtest();




## Plotting.




function plot_objective_function(f, limits = LIMITS; bins=100, kwargs...)
	clf()

    rx = range(limits.X[1], stop=limits.X[2], length=bins)
    ry = range(limits.Y[1], stop=limits.Y[2], length=bins)

    # if USE_PGFPLOTS
    #     a = Axis(Plots.Image((x,y)->sierra([x,y], η=η, decay=(i <= 3)), (-15,15), (-15,15),
    #             xbins=201, ybins=201, colormap=viridis_r, colorbar=(i % 3 == 0)),
    #         height="8cm", width="8cm",
    #         style=(i <= 3) ? "xmajorticks=false" : "",
    #         title="\$\\eta=$η\$, decay=$(𝕀(i <= 3))",)

    # note reverse and x, y switch (imshow has a weird mapping)
    imshow([f([x,y]; kwargs...) for y in reverse(ry), x in rx],
        extent=[limits.X[1], limits.X[2], limits.Y[1], limits.Y[2]],
        cmap="viridis_r")
    # display(gcf())
end


function plot_cem(P, samples, elite)
	if !isempty(samples)
		# Plotting
	    gca().collections = [] # update figure instead of redrawing
	    scatter(samples[1,:], samples[2,:], s=7, c="black", edgecolors="white", linewidths=1/2)
	    scatter(elite[1,:], elite[2,:], s=7, c="red", edgecolors="white", linewidths=1/2)
	    # scatter(samples[1,order[1:m_elite]], samples[2,order[1:m_elite]], s=7, c="red", edgecolors="white", linewidths=1/2)
	    xlim(LIMITS.X)
	    ylim(LIMITS.Y)
	end

    plot_distribution(P, false)
    # display(gcf())
end


function plot_distribution(P, clear=true)
    # Plot covariance.
    if clear
	    gca().collections = [] # update figure instead of redrawing
	end
    X = range(LIMITS.X[1], stop=LIMITS.X[2], length=100)
    Y = range(LIMITS.Y[1], stop=LIMITS.Y[2], length=100)
    Z = [pdf(P, [x,y]) for y in Y, x in X] # Note: reverse in X, Y.
    contour(Z, extent=[LIMITS.X[1], LIMITS.X[2], LIMITS.Y[1], LIMITS.Y[2]], cmap="hot", alpha=0.3)
    display(gcf())
    sleep(1/10)
end


# AST.playback(mdp, A, sim->(plot_objective_function(sim.f), plot_cem(sim.distribution, sim.samples, sim.elite)))
# AST.playback(mdp, A, sim_playback_plot)
function sim_playback_plot(sim)
	plot_objective_function(sim.f)
	plot_cem(sim.distribution, sim.samples, sim.elite)
end



#=
MWE:

include("test\\CEMTest.jl")

A = AST.playout(mdp, planner)
# or
A = collect(keys(mdp.top_paths))[1]

AST.playback(mdp, A, sim_playback_plot)

=#