# using Revise
using POMDPStressTesting
using Distributions
using Parameters
using LinearAlgebra


@with_kw mutable struct Walk2DParams
    startx::Float64 = 0 # Starting x-position
    starty::Float64 = 0 # Starting y-position
    threshx::Float64 = 10 # +- x boundary threshold
    threshy::Float64 = 10 # +- y boundary threshold
    endtime::Int64 = 30 # Simulate end time
    corners::Matrix = [threshx  threshy;
                      -threshx  threshy;
                       threshx -threshy;
                      -threshx -threshy]
end


# Implement abstract GrayBox.Simulation
@with_kw mutable struct Walk2DSim <: GrayBox.Simulation
    params::Walk2DParams = Walk2DParams() # Parameters
    x::Float64 = 0 # Current x-position
    y::Float64 = 0 # Current x-position
    t::Int64 = 0 # Current time
    x_distribution::Distribution = Normal(0, 1) # x transition distribution
    y_distribution::Distribution = Normal(0, 1) # y transition distribution
end


# Override from GrayBox
GrayBox.environment(sim::Walk2DSim) = GrayBox.Environment(:x => sim.x_distribution, :y => sim.y_distribution)


# Override from GrayBox (NOTE: used with ASTSeedAction)
function GrayBox.transition!(sim::Walk2DSim)
    # We sample the environment and apply the transition
    environment::GrayBox.Environment = GrayBox.environment(sim) # Get the environment distributions
    sample::GrayBox.EnvironmentSample = rand(environment) # Sample from the environment
    sim.t += 1 # Keep track of time
    sim.x += sample[:x].value # Move agent using sampled x-value
    sim.y += sample[:y].value # Move agent using sampled y-value
    return logpdf(sample)::Real # Summation handled by `logpdf()`
end


# Override from GrayBox (NOTE: used with ASTSampleAction)
function GrayBox.transition!(sim::Walk2DSim, sample::GrayBox.EnvironmentSample)
    # The environment was sampled for us, and we just apply the transition
    sim.t += 1 # Keep track of time
    sim.x += sample[:x].value # Move agent using sampled value from input
    sim.y += sample[:y].value # Move agent using sampled value from input
    return logpdf(sample)::Real # Summation handled by `logpdf()`
end


# Override from BlackBox
function BlackBox.initialize!(sim::Walk2DSim)
    sim.t = 0
    sim.x = sim.params.startx
    sim.y = sim.params.starty
end


# Override from BlackBox
BlackBox.distance!(sim::Walk2DSim) = minimum(mapslices(corner->norm([sim.x, sim.y] - corner), sim.params.corners, dims=2))


# Override from BlackBox
BlackBox.isevent!(sim::Walk2DSim) = abs(sim.x) >= sim.params.threshx && abs(sim.y) >= sim.params.threshy


# Override from BlackBox
BlackBox.isterminal!(sim::Walk2DSim) = BlackBox.isevent!(sim) || sim.t >= sim.params.endtime


# Override from BlackBox (NOTE: used with ASTSeedAction)
function BlackBox.evaluate!(sim::Walk2DSim)
    logprob::Real  = GrayBox.transition!(sim) # Step simulation
    distance::Real = BlackBox.distance!(sim) # Calculate miss distance
    event::Bool    = BlackBox.isevent!(sim) # Check event indication
    return (logprob::Real, distance::Real, event::Bool)
end


# Override from BlackBox (NOTE: used with ASTSampleAction)
function BlackBox.evaluate!(sim::Walk2DSim, sample::GrayBox.EnvironmentSample)
    logprob::Real  = GrayBox.transition!(sim, sample) # Step simulation given input sample
    distance::Real = BlackBox.distance!(sim) # Calculate miss distance
    event::Bool    = BlackBox.isevent!(sim) # Check event indication
    return (logprob::Real, distance::Real, event::Bool)
end


function setup_ast(seed=AST.DEFAULT_SEED; solver=PPOSolver)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = Walk2DSim()

    # AST MDP formulation object
    # NOTE: Use either {ASTSeedAction} or {ASTSampleAction}
    mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim) # ASTSampleAction for use with DRL solvers (TRPO/PPO)
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10 # record top k best trajectories
    mdp.params.seed = seed # set RNG seed for determinism
    n_iterations = 1000 # number of algorithm iterations

    # Choose a solver
    if solver == RandomSearchSolver
        solver = RandomSearchSolver(depth=sim.params.endtime,
                                    n_iterations=n_iterations)
    elseif solver == MCTSASTSolver
        solver = MCTSASTSolver(depth=sim.params.endtime, # tree depth
                               exploration_constant=1.0, # UCT exploration
                               k_action=1.0, # action widening
                               alpha_action=0.5, # action widening
                               n_iterations=n_iterations)
    elseif solver == TRPOSolver
        solver = TRPOSolver()
    elseif solver == PPOSolver
        solver = PPOSolver()
    end

    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return (planner, mdp::ASTMDP)
end


function run_ast(seed=AST.DEFAULT_SEED; kwargs...)
    (planner, mdp) = setup_ast(seed; kwargs...)

    action_trace::Vector{ASTAction} = playout(mdp, planner) # work done here.
    final_state::ASTState = playback(mdp, action_trace, sim->(sim.x, sim.y))
    failure_rate::Float64 = print_metrics(mdp)

    return mdp::ASTMDP, action_trace::Vector{ASTAction}, failure_rate::Float64
end


(mdp, action_trace, failure_rate) = run_ast()


using PyPlot
function plot_trace(mdp, k=mdp.params.top_k)
    trace = playback(mdp, get_top_path(mdp, k), sim->[sim.x, sim.y]; return_trace=true)
    x = map(first, trace)
    y= map(last, trace)
    plot(x, y)
end


nothing # Suppress REPL