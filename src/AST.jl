"""
Provides virtual interface for Adaptive Stress Testing (AST) formulation of MDPs/POMDPs
"""
module AST

include("BlackBox.jl")
include("RandomSeedGenerator.jl")

using .RandomSeedGenerator
using Random
using Parameters
using DataStructures
using POMDPPolicies
using POMDPSimulators
using POMDPs

export
    BlackBox,

    Params,
    ASTMDP,
    ASTState,
    ASTAction,

    reward,
    action,
    initialize,
    parameters, # TODO: remove
    initial_seed, # TODO: remove
    simulate,
    isterminal,

    playback

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
    # @warn "A[1] reinstate..." # TODO
    if length(A) > 0
        h = hash(A[1])
        for i in 2:length(A)
            h = hash(h, hash(A[i]))
        end
    else
        h = hash([])
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


# TODO: put in separate module
mutable struct Tracker
    actions::Vector{ASTAction}
    q_values::Vector{Float64}
    q_values_rev::Vector{Float64}

    Tracker() = new(ASTAction[], Float64[], Float64[])
end

function Base.empty!(tracker::Tracker)
    empty!(tracker.actions)
    empty!(tracker.q_values)
    empty!(tracker.q_values_rev)
end

Base.hash(tracker::Tracker) = hash(tracker.actions)
Base.:(==)(t1::Tracker, t2::Tracker) = t1.actions == t2.actions

push_action!(tracker::Tracker, a::ASTAction) = push!(tracker.actions, a)
push_q_value!(tracker::Tracker, q::Float64) = push!(tracker.q_values, q)
push_q_value_rev!(tracker::Tracker, q::Float64) = push!(tracker.q_values_rev, q)
append_actions!(tracker::Tracker, a::Vector) = append!(tracker.actions, a)
append_q_values!(tracker::Tracker, q::Vector) = append!(tracker.q_values, q)

function combine_q_values!(tracker::Tracker)
    if !isempty(tracker.q_values_rev)
        append!(tracker.q_values, reverse(tracker.q_values_rev))
        empty!(tracker.q_values_rev)
    end
end

get_actions(tracker::Tracker) = tracker.actions
get_q_values(tracker::Tracker) = tracker.q_values


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

    # TODO: {Tracker, Float64}...
    top_paths::PriorityQueue{Any, Float64} # Collection of best paths in the tree # TODO: MCTS specific.
    tracker::Tracker

    function ASTMDP(params::Params, sim)
        rsg::RSG = RSG(params.rsg_length, params.init_seed)
        top_paths = PriorityQueue{Any, Float64}(Base.Order.Forward)
        tracker = Tracker()
        return new(params, sim, hash(0), 1, rsg, deepcopy(rsg), !isnothing(params.reset_seed) ? RSG(params.rsg_length, params.reset_seed) : nothing, top_paths, tracker)
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


# TODO: Handle `rng` (UNUSED)
function POMDPs.initialstate(mdp::ASTMDP, rng::AbstractRNG)
    mdp.t_index = 1
    BlackBox.initialize(mdp.sim)

    if !isnothing(mdp.reset_rsg)
        # If resetting RSG is specified
        mdp.rsg = deepcopy(mdp.reset_rsg)
    end

    s::ASTState = ASTState(mdp.t_index, nothing, ASTAction(deepcopy(mdp.initial_rsg)))
    mdp.sim_hash = s.hash
    return s::ASTState
end


# TODO: Handle `rng`
# Generate next state for AST
# function POMDPs.gen(::DDNOut{(:sp, :r)}, mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG)
function POMDPs.gen(mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG)
    @assert mdp.sim_hash == s.hash
    mdp.t_index += 1 # TODO: Why? What is t_index doing?
    # Random.seed!(a.rsg)
    set_global_seed(a.rsg)

    # Step black-box simulation
    (prob::Float64, isevent::Bool, miss_distance::Float64) = BlackBox.evaluate(mdp.sim)

    sp = ASTState(mdp.t_index, s, a) # TODO: What's going on?
    mdp.sim_hash = sp.hash
    r::Float64 = reward(mdp, prob, isevent, BlackBox.isterminal(mdp.sim), miss_distance)

    return (sp=sp, r=r)
end


# TODO: go_to_state.... ?


function POMDPs.isterminal(mdp::ASTMDP, s::ASTState)
    return BlackBox.isterminal(mdp.sim)
end


POMDPs.discount(mdp::ASTMDP) = 1.0 # 0.9 # TODO: ?


# TODO: POMDPs.action # i.e. random_action
function POMDPs.action(policy::RandomPolicy, s::ASTState)
    rsg::RSG = policy.problem.rsg
    next!(rsg)
    return ASTAction(deepcopy(rsg))
end

# TODO: blend with POMDPs.action
random_action(mdp::ASTMDP, ::ASTState, dpw) = random_action(mdp)
function random_action(mdp::ASTMDP)
    rsg::RSG = mdp.rsg
    next!(rsg)
    return ASTAction(deepcopy(rsg))
end

function POMDPs.actions(mdp::ASTMDP)
    # actions = ASTAction[]
    # k = 10 # TODO: Not what 'k' is for.
    # for i in 1:k
    #     rsg::RSG = mdp.rsg
    #     next!(rsg)
    #     push!(actions, ASTAction(deepcopy(rsg)))
    # end
    # return actions::Vector{ASTAction}
    # return [ASTAction(mdp.rsg)]
    # TODO: Make bigger?
    return [random_action(mdp) for k in 1:10]
end


function go_to_state(mdp::ASTMDP, target_state::ASTState)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    actions = get_action_sequence(target_state)
    R = 0.0
    for a in actions
        s, r = gen(mdp, s, a, Random.GLOBAL_RNG)
        R += r
        @show s.parent
    end
    @assert s == target_state
    return (R, actions)
end


function rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        a::ASTAction = random_action(mdp)

        push_action!(mdp.tracker, a)

        (sp, r) = gen(DDNOut(:sp, :r), mdp, s, a, Random.GLOBAL_RNG)
        # @show sp, "NEXT STATE: s′"

        q_value = r + discount(mdp)*rollout(mdp, sp, d-1)
        push_q_value_rev!(mdp.tracker, q_value)
        return q_value
    end
end



function get_action_sequence(s::ASTState)
    actions::Vector{ASTAction} = ASTAction[]

    # Trace up the tree
    while !isnothing(s.parent)
        prepend!(actions, s.action)
        s = s.parent
    end

    return actions::Vector{ASTAction}
end




function trace!(s_current::ASTState, snode)
    q_values = Float64[]
    # if !haskey()  # TODO
    s = s_current
    while !isnothing(s.parent)
        q = snode.tree.q[snode.tree.s_lookup[s.parent]]
        push!(q_values, q)
        s = s.parent
    end
    return reverse(q_values)
end


# i.e. selectAction, actionSelection (find best)
function next_action(mdp::ASTMDP, s::ASTState, snode)
    n = 100 # TODO: Parameterize
    top_k = 10 # TODO: Parameterize
    simulator = RolloutSimulator(max_steps=mdp.params.max_steps)
    policy = RandomPolicy(mdp) # uses the POMDPs.actions function
    R = 0.0
    for i in 1:n
        (R, actions) = go_to_state(mdp, s)

        # TODO: tracker?
        # @show actions, R
        # empty!(mdp.tracker)
        # append_actions!(mdp.tracker, actions)
        # q_values = trace!(s, snode)
        # append_q_values!(mdp.tracker, q_values)

        # R += simulate(simulator, mdp, policy)
        # TODO: HistoryRecorder??

        hr = HistoryRecorder()
        history = simulate(hr, mdp, policy)
        R = sum(map(h->h[:r], history))

        # TODO: queue! tracker
        # TODO: deepcopy parameters
        # TODO: Abstract this to separate module

        # combine_q_values!(mdp.tracker)

        # if !haskey(mdp.top_paths, mdp.tracker) # TODO: haskey # error
        actions = map(h->h[:a], history)
        # if !haskey(mdp.top_paths, history)
        if !haskey(mdp.top_paths, actions)
            # enqueue!(mdp.top_paths, deepcopy(mdp.tracker), R)
            enqueue!(mdp.top_paths, actions, R)
            # enqueue!(mdp.top_paths, history, R)
            while length(mdp.top_paths) > top_k
                dequeue!(mdp.top_paths)
            end
        else
            @show R, "HERE"
        end

        # TODO: cpu timing cut-off
    end

    # TODO: track actions and q_values internal to MCTS ?

    # @show R

    go_to_state(mdp, s) # Leave simulation in the current state

    # Extract actions taken at current state
    # display(snode)

    # a = action(planner, s) # TODO: Might cause StackOverflow...

    return random_action(mdp)
    # return a # TODO: return best action
end

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




# function stress_test(mdp::ASTMDP)
#     simulator = RolloutSimulator(max_steps=mdp.params.max_steps)
#     policy = RandomPolicy(mdp) # uses the POMDPs.actions function
#     r = simulate(simulator, mdp, policy)
# end


function playback(mdp::ASTMDP)
    k = collect(keys(mdp.top_paths))
    actions = k[end]

    rng = Random.GLOBAL_RNG # Not used.
    s = initialstate(mdp, rng)
    BlackBox.initialize(mdp.sim)
    @show mdp.sim.x
    for a in actions
        (sp, r) = gen(mdp, s, a, rng)
        s = sp
        @show mdp.sim.x
    end
end

end  # module AST