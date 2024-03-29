# using Revise
using POMDPStressTesting
using Distributions
using Parameters


@with_kw mutable struct Walk1DParams
    startx::Float64 = 0 # Starting x-position
    threshx::Float64 = 10 # +- boundary threshold
    endtime::Int64 = 30 # Simulate end time
end


# Implement abstract GrayBox.Simulation
@with_kw mutable struct Walk1DSim <: GrayBox.Simulation
    params::Walk1DParams = Walk1DParams() # Parameters
    x::Float64 = 0 # Current x-position
    t::Int64 = 0 # Current time
    distribution::Distribution = Normal(0, 1) # Transition distribution
end


# Override from GrayBox
GrayBox.environment(sim::Walk1DSim) = GrayBox.Environment(:x => sim.distribution)


# Override from GrayBox (NOTE: used with ASTSeedAction)
function GrayBox.transition!(sim::Walk1DSim)
    # We sample the environment and apply the transition
    environment::GrayBox.Environment = GrayBox.environment(sim) # Get the environment distributions
    sample::GrayBox.EnvironmentSample = rand(environment) # Sample from the environment
    sim.t += 1 # Keep track of time
    sim.x += sample[:x].value # Move agent using sampled value from input
    return logpdf(sample)::Real # Summation handled by `logpdf()`
end


# Override from GrayBox (NOTE: used with ASTSampleAction)
function GrayBox.transition!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    # The environment was sampled for us, and we just apply the transition
    sim.t += 1 # Keep track of time
    sim.x += sample[:x].value # Move agent using sampled value from input
    return logpdf(sample)::Real # Summation handled by `logpdf()`
end


# Override from BlackBox
function BlackBox.initialize!(sim::Walk1DSim)
    sim.t = 0
    sim.x = sim.params.startx
end


# Override from BlackBox
BlackBox.distance(sim::Walk1DSim) = max(sim.params.threshx - abs(sim.x), 0)


# Override from BlackBox
BlackBox.isevent(sim::Walk1DSim) = abs(sim.x) >= sim.params.threshx


# Override from BlackBox
BlackBox.isterminal(sim::Walk1DSim) = BlackBox.isevent(sim) || sim.t >= sim.params.endtime


# Override from BlackBox (NOTE: used with ASTSeedAction)
function BlackBox.evaluate!(sim::Walk1DSim)
    logprob::Real = GrayBox.transition!(sim) # Step simulation
    d::Real = BlackBox.distance(sim) # Calculate miss distance
    event::Bool = BlackBox.isevent(sim) # Check event indication
    return (logprob::Real, d::Real, event::Bool)
end


# Override from BlackBox (NOTE: used with ASTSampleAction)
function BlackBox.evaluate!(sim::Walk1DSim, sample::GrayBox.EnvironmentSample)
    logprob::Real = GrayBox.transition!(sim, sample) # Step simulation given input sample
    d::Real = BlackBox.distance(sim) # Calculate miss distance
    event::Bool = BlackBox.isevent(sim) # Check event indication
    return (logprob::Real, d::Real, event::Bool)
end


# Explicit state (optional)
GrayBox.state(sim::Walk1DSim) = [sim.x]


function setup_ast(seed=AST.DEFAULT_SEED; solver=TRPOSolver)
    # Create gray-box simulation object
    sim::GrayBox.Simulation = Walk1DSim()

    # AST MDP formulation object
    # NOTE: Use either {ASTSeedAction} or {ASTSampleAction} (when using TRPO/PPO/CEM, use ASTSampleAction)
    if solver in [TRPOSolver, PPOSolver, CEMSolver]
        mdp::ASTMDP = ASTMDP{ASTSampleAction}(sim)
    else
        mdp = ASTMDP{ASTSeedAction}(sim)
    end
    mdp.params.debug = true # record metrics
    mdp.params.top_k = 10 # record top k best trajectories
    mdp.params.seed = seed # set RNG seed for determinism
    n_iterations = 1000 # number of algorithm iterations

    # Choose a solver (examples of each)
    if solver == RandomSearchSolver
        solver = RandomSearchSolver(n_iterations=n_iterations,
                                    episode_length=sim.params.endtime)
    elseif solver == MCTSPWSolver
        solver = MCTSPWSolver(n_iterations=n_iterations,
                              exploration_constant=1.0, # UCT exploration
                              k_action=1.0, # action widening
                              alpha_action=0.5, # action widening
                              depth=sim.params.endtime) # tree depth (i.e. episode length)
    elseif solver == CEMSolver
        solver = CEMSolver(n_iterations=n_iterations,
                           episode_length=sim.params.endtime)
    elseif solver == TRPOSolver
        solver = TRPOSolver(num_episodes=n_iterations,
                            episode_length=sim.params.endtime)
    elseif solver == PPOSolver
        solver = PPOSolver(num_episodes=n_iterations,
                           episode_length=sim.params.endtime)
    end

    # Get online planner (no work done, yet)
    planner = solve(solver, mdp)

    return planner
end


function run_ast(seed=AST.DEFAULT_SEED; kwargs...)
    planner = setup_ast(seed; kwargs...)

    action_trace::Vector{ASTAction} = search!(planner) # work done here
    final_state::ASTState = playback(planner, action_trace, sim->sim.x)
    failure_rate = print_metrics(planner)

    return planner, action_trace::Vector{ASTAction}, failure_rate
end

(planner, action_trace, failure_rate) = run_ast()

nothing # Suppress REPL