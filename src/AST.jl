"""
Provides virtual interface for Adaptive Stress Testing (AST) formulation of MDPs/POMDPs
"""
module AST

include("BlackBox.jl")

using Parameters
using POMDPs

export
    BlackBox,

    Params,
    Simulation,

    reward,
    action,
    initialize,
    parameters, # TODO: remove
    initial_seed, # TODO: remove
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
    BlackBox.evaluate (or: step, update)

TODO:
    [] Specific AST struct???
        [] POMDP/MDP formulation of AST as ::MDP{S,A}
    [] Integrate with solvers (i.e. MCTS.jl with "single" progressive widening)
    [] @impl_dep: implementation dependencies (see pomdp.jl for example)
    [] Benchmarking/Results tools
    [] BlackBox.evaluate vs. BlackBox.step
=#

const DEFAULT_RSG_LENGTH = 3 # Default random seed generator (RSG) length (TODO)

# TODO: RSG
# Credit: Ritchie Lee

using Random
using Base.Iterators
using IterTools

struct RSG
    state::Vector{UInt32}
end
RSG(len::Int64=1, seed::Int64=0) = seed_to_state_itr(len, seed) |> collect |> RSG

set_from_seed!(rsg::RSG, len::Int64, seed::Int64) = copy!(rsg.state, seed_to_state_itr(len, seed))
seed_to_state_itr(len::Int64, seed::Int64) = take(iterated(hash_uint32, seed), len)
hash_uint32(x) = UInt32(hash(x) & 0x00000000FFFFFFFF) # Take lower 32-bits


"""
    AST.Params

Adaptive Stress Testing specific simulation parameters.
"""
@with_kw mutable struct Params
    max_steps::Int64 = 0 # Maximum simulation time step (for runaways in simulation)
    rsg_length::Int64 = 3 # Dictates number of unique available random seeds ((????))
    init_seed::Int64 = 0 # Initial seed value
    reset_seed::Union{Nothing, Int64} = nothing # Reset to this seed value on initialize()
end
Params(max_steps::Int64, rsg_length::Int64, init_seed::Int64) = Params(max_steps=max_steps, rsg_length=rsg_length, init_seed=init_seed)




mutable struct ASTAction
    # TODO
    rsg::RSG

    ASTAction(rsg::RSG) = new(rsg)
    ASTAction(len::Int64=DEFAULT_RSG_LENGTH, seed::Int64=0) = ASTAction(RSG(len, seed))
end


mutable struct ASTState
    # TODO
    t_index::Int64 # Sanity check that time corresponds
    hash::UInt64 # Hash simulation state to match with ASTState
    parent::Union{Nothing,ASTState} # Parent state, `nothing` if root
    action::ASTAction # Action taken from parent, 0 if root
end

# TODO: Put into ASTState struct contructor
function ASTState(t_index::Int64, parent::Union{Nothing,ASTState}, action::ASTAction)
    s = ASTState(t_index, 0, parent, action)
    s.hash = hash(s)
    return s
end


# TODO: Understand. Rewrite.
Base.hash(a::ASTAction) = hash(a.rsg)
function Base.hash(A::Vector{ASTAction})
    h = hash(A[1])
    for i in 2:length(A)
        h = hash(h, hash(A[i]))
    end
    return h
end

# TODO: Understand. Rewrite.
function Base.hash(s::ASTState)
    h = hash(s.t_index)
    h = hash(h, hash(isnothing(s.parent) ? nothing : s.parent.hash))
    h = hash(h, hash(s.action))
    return h
end

# Helpers. Credit Ritchie Lee
Base.:(==)(w::ASTAction,v::ASTAction) = w.rsg == v.rsg
Base.:(==)(w::ASTState,v::ASTState) = hash(w) == hash(v)
Base.isequal(w::ASTAction,v::ASTAction) = isequal(w.rsg,v.rsg)
Base.isequal(w::ASTState,v::ASTState) = hash(w) == hash(v)


# TODO: Naming
mutable struct ASTMDP <: MDP{ASTState, ASTAction}
    # TODO
    params::Params # AST simulation parameters
    sim::BlackBox.Simulation # Black-box simulation struct
    sim_hash::UInt64 # Hash to keep simulations in sync

    t_index::Int64 # TODO
    rsg::RSG # Random seed generator
    initial_rsg::RSG # Initial random seed generator
    reset_rsg::Union{Nothing,RSG} # Reset to this RSG if provided

    function ASTMDP(params::Params, sim)
        rsg::RSG = RSG(params.rsg_length, params.init_seed)
        return new(params, sim, hash(0), 1, rsg, deepcopy(rsg), !isnothing(params.reset_seed) ? RSG(params.rsg_length, params.reset_seed) : nothing)
    end
end



#=
"""
    AST.Simulation

Adaptive Stress Testing simulation object.
"""
@with_kw mutable struct Simulation
    params::Params # AST simulation parameters
    sim::BlackBox.Simulation # Black-box simulation struct
    sim_hash::UInt64 # Hash to keep simulations in sync

    # Function defined by BlackBox virtual interface.
    # get_reward # TODO

    t_index::Int64 # Starts at 1 and counts up (???)
    rsg # ::RSG # TODO
end
=#




"""
    reward(s::State)::Float64

Reward function for the AST formulation. Defaults to:

    0               s ∈ Event            # Terminates with event, maximum reward of 0
    -∞,             s ̸∈ Event and t ≥ T  # Terminates without event, maximum negative reward of -∞
    log P(s′ | s),  s ̸∈ Event and t < T  # Each non-terminal step, accumulate reward correlated with the transition probability
"""
# function reward end
# function POMDPs.reward(mdp::ASTMDP, s::ASTState, a::ASTAction, sp::ASTState)
function POMDPs.reward(mdp::ASTMDP, prob::Float64, isevent::Bool, isterminal::Bool, miss_distance::Float64)
    r = log(prob)
    if isevent
        r += 0.0
    elseif isterminal # Incur miss distance cost only if !isevent && isterminal
        r += -miss_distance
    end
    return r
end


# TODO: Handle `rng`
function POMDPs.initialstate(mdp::ASTMDP, rng::AbstractRNG)
    mdp.t_index = 1
    BlackBox.initialize(mdp.sim)

    if !isnothing(mdp.reset_rsg)
        # If resetting RSG is specified
        mdp.rsg = deepcopy(ast.reset_rsg)
    end

    s::ASTState = ASTState(mdp.t_index, nothing, ASTAction(deepcopy(mdp.initial_rsg)))
    mdp.sim_hash = s.hash
    return s::ASTState
end


# TODO: Handle `rng`
# Generate next state for AST
function gen(::DDNOut{(:sp, :r)}, mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG)
    @assert mdp.sim_hash == s.hash
    mdp.t_index += 1 # TODO: Why? What is t_index doing?
    Random.seed!(a.rsg.state) # TODO: import Random.seed!; Random.seed!(a::ASTAction) = Random.seed!(a.rsg.state)

    # Step black-box simulation
    (prob::Float64, isevent::Bool, miss_distance::Float64) = BlackBox.evaluate(mdp.sim)

    sp = ASTState(mdp.t_index, s, a) # TODO: What's going on?
    mdp.sim_hash = sp.hash
    r::Float64 = reward(mdp, prob, isevent, BlackBox.isterminal(mdp.sim), miss_distance)

    return (sp=sp, r=r)
end


# TODO: go_to_state....


function POMDPs.isterminal(mdp::ASTMDP, s::ASTState)
    return BlackBox.isterminal(mdp.sim)
end


function POMDPs.transition(mdp::ASTMDP, s::ASTState, a::ASTAction)
    # TODO
    return 1.0
end

POMDPs.discount(mdp::ASTMDP) = 0.9 # TODO: ?


# TODO: POMDPs.action # i.e. random_action


# """
#     action()::Seed

# Returns new seed as the action.
# """
# function action end


# """
#     parameters()

# Control the parameters used by AST.
# """
# function parameters end


# """
#     initial_seed(a::Seed)

# Control the initial seed used for the RNG.
# """
# function initial_seed end


"""
    simulate()

Run AST simulation.
"""
function simulate end


end  # module AST