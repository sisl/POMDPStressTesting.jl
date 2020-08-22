using POMDPStressTesting
using Distributions
using Parameters


@with_kw mutable struct Walk1DParams
    startx::Float64 = 0 # Starting x-position
    threshx::Float64 = 10 # +- boundary threshold
    endtime::Int64 = 30 # Simulate end time
end


# Implement abstract BlackBox.Simulation
@with_kw mutable struct Walk1DSim <: BlackBox.Simulation
    params::Walk1DParams = Walk1DParams() # Parameters
    x::Float64 = 0 # Current x-position
    t::Int64 = 0 # Current time
    distribution::Distribution = Normal(0, 1) # Transition model
end


# Override from BlackBox
function BlackBox.initialize!(sim::Walk1DSim)
    sim.t = 0
    sim.x = sim.params.startx
end


# Override from BlackBox
function BlackBox.transition!(sim::Walk1DSim)
    sim.t += 1 # Keep track of time
    sample = rand(sim.distribution) # Sample value from distribution
    logprob = logpdf(sim.distribution, sample) # Get log-probability of sample
    sim.x += sample # Move agent
    return logprob::Real
end


# Override from BlackBox
BlackBox.distance!(sim::Walk1DSim) = max(sim.params.threshx - abs(sim.x), 0)


# Override from BlackBox
BlackBox.isevent!(sim::Walk1DSim) = abs(sim.x) >= sim.params.threshx


# Override from BlackBox
BlackBox.isterminal!(sim::Walk1DSim) = BlackBox.isevent!(sim) || sim.t >= sim.params.endtime


# Override from BlackBox
function BlackBox.evaluate!(sim::Walk1DSim)
    logprob::Real  = BlackBox.transition!(sim) # Step simulation
    distance::Real = BlackBox.distance!(sim) # Calculate miss distance
    event::Bool    = BlackBox.isevent!(sim) # Check event indication
    return (logprob::Real, distance::Real, event::Bool)
end


function setup_ast()
    # Create black-box simulation object
    sim::BlackBox.Simulation = Walk1DSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP(sim)
    mdp.params.debug = true # record metrics
    mdp.params.seed = 1
    mdp.params.top_k = 10

    # Hyperparameters for MCTS-PW as the solver
    solver = MCTSASTSolver(depth=sim.params.endtime,
                           exploration_constant=10.0,
                           k_action=0.1,
                           alpha_action=0.85,
                           n_iterations=1000)

    policy = solve(solver, mdp)

    return (policy, mdp, sim)
end


(policy, mdp, sim) = setup_ast()

action_trace = playout(mdp, policy)

final_state = playback(mdp, action_trace, sim->sim.x)

print_metrics(mdp)

nothing # Suppress REPL