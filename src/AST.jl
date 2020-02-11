"""
Provides virtual interface for Adaptive Stress Testing (AST) formulation of MDPs/POMDPs
"""
module AST

export
    reward,
    action,
    parameters,
    initialize,
    initial_seed,
    simulate,
    isterminal

#=
Define interface to follow (AST. or POMDPStressTesting.):
    AST.reward (defaults to logprob of T, event e, miss distance d), make it customizable.
    AST.actions # seeds
    AST.parameters
    AST.simulate
    AST.initial_seed

Define "black-box" interface (separate this from AST formulation):
    BlackBox.initialize
    BlackBox.isevent
    BlackBox.miss_distance
    BlackBox.transition_prob
    BlackBox.evaluate (or step)

TODO:
    [] Specific AST struct???
        [] POMDP/MDP formulation of AST as ::MDP{S,A}
    [] Integrate with solvers (i.e. MCTS.jl with "single" progressive widening)
    [] @impl_dep: implementation dependencies (see pomdp.jl for example)
    [] Benchmarking/Results tools
    [] BlackBox.evaluate vs. BlackBox.step
=#

"""
    reward(s::State)::Float64

Reward function for the AST formulation. Defaults to:

    0               s ∈ Event            # Terminates with event, maximum reward of 0
    -∞,             s ̸∈ Event and t ≥ T  # Terminates without event, maximum negative reward of -∞
    log P(s′ | s),  s ̸∈ Event and t < T  # Each non-terminal step, accumulate reward correlated with the transition probability
"""
function reward end


"""
    action()::Seed

Returns new seed as the action.
"""
function action end


"""
    parameters()

Control the parameters used by AST.
"""
function parameters end


"""
    initial_seed(a::Seed)

Control the initial seed used for the RNG.
"""
function initial_seed end


"""
    simulate()

Run AST simulation.
"""
function simulate end


end  # module AST