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
    q_value::Float64 # Saved Q-value
end

# TODO: Put into ASTState struct contructor
function ASTState(t_index::Int64, parent::Union{Nothing,ASTState}, action::ASTAction)
    s = ASTState(t_index, UInt64(0), parent, action, 0.0)
    s.hash = hash(s)
    return s
end

function ASTState(t_index::Int64, action::ASTAction)
    return ASTState(t_index, nothing, action)
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
function POMDPs.initialstate(mdp::ASTMDP, rng::AbstractRNG=Random.GLOBAL_RNG)
    mdp.t_index = 1
    BlackBox.initialize(mdp.sim)

    if !isnothing(mdp.reset_rsg)
        # If resetting RSG is specified
        mdp.rsg = deepcopy(mdp.reset_rsg)
    end

    s::ASTState = ASTState(mdp.t_index, ASTAction(deepcopy(mdp.initial_rsg)))
    mdp.sim_hash = s.hash
    return s::ASTState
end


# TODO: Handle `rng`
# Generate next state for AST
function POMDPs.gen(mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG)
    @assert mdp.sim_hash == s.hash
    mdp.t_index += 1
    set_global_seed(a.rsg)

    # Step black-box simulation
    (prob::Float64, isevent::Bool, miss_distance::Float64) = BlackBox.evaluate(mdp.sim)

    # Update state
    sp = ASTState(mdp.t_index, s, a)
    mdp.sim_hash = sp.hash
    r::Float64 = reward(mdp, prob, isevent, BlackBox.isterminal(mdp.sim), miss_distance)
    sp.q_value = r

    return (sp=sp, r=r)
end


# TODO: go_to_state.... ?


function POMDPs.isterminal(mdp::ASTMDP, s::ASTState)
    # @assert mdp.sim_hash == s.hash
    if mdp.sim_hash != s.hash
        # @show mdp
        # @show s
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff11
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff22
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff33
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff44
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff55
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff66
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff77
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff88
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff99
        # TODO IMPORTANT: @warn "isterminal FIX" #ff1110
        # TODO IMPORTANT: @warn "isterminal FIX" #ff2211
        # TODO IMPORTANT: @warn "isterminal FIX" #ff3312
        # TODO IMPORTANT: @warn "isterminal FIX" #ff4413
        # TODO IMPORTANT: @warn "isterminal FIX" #ff5514
        # TODO IMPORTANT: @warn "isterminal FIX" #ff6615
        # TODO IMPORTANT: @warn "isterminal FIX" #ff7716
        # TODO IMPORTANT: @warn "isterminal FIX" #ff8817
        # TODO IMPORTANT: @warn "isterminal FIX" #ff9918
        # TODO IMPORTANT: @warn "isterminal FIX" #f10019
        # TODO IMPORTANT: @warn "isterminal FIX" #f11120
        # TODO IMPORTANT: @warn "isterminal FIX" #f12221
        # TODO IMPORTANT: @warn "isterminal FIX" #f13322
        # TODO IMPORTANT: @warn "isterminal FIX" #f14423
        # TODO IMPORTANT: @warn "isterminal FIX" #f15524
        # TODO IMPORTANT: @warn "isterminal FIX" #f16625
        # TODO IMPORTANT: @warn "isterminal FIX" #f17726
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff11
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff22
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff33
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff44
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff55
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff66
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff77
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff88
        # TODO IMPORTANT: @warn "isterminal FIX" #ffff99
        # TODO IMPORTANT: @warn "isterminal FIX" #ff1110
        # TODO IMPORTANT: @warn "isterminal FIX" #ff2211
        # TODO IMPORTANT: @warn "isterminal FIX" #ff3312
        # TODO IMPORTANT: @warn "isterminal FIX" #ff4413
        # TODO IMPORTANT: @warn "isterminal FIX" #ff5514
        # TODO IMPORTANT: @warn "isterminal FIX" #ff6615
        # TODO IMPORTANT: @warn "isterminal FIX" #ff7716
        # TODO IMPORTANT: @warn "isterminal FIX" #ff8817
        # TODO IMPORTANT: @warn "isterminal FIX" #ff9918
        # TODO IMPORTANT: @warn "isterminal FIX" #f10019
        # TODO IMPORTANT: @warn "isterminal FIX" #f11120
        # TODO IMPORTANT: @warn "isterminal FIX" #f12221
        # TODO IMPORTANT: @warn "isterminal FIX" #f13322
        # TODO IMPORTANT: @warn "isterminal FIX" #f14423
        # TODO IMPORTANT: @warn "isterminal FIX" #f15524
        # TODO IMPORTANT: @warn "isterminal FIX" #f16625
        # TODO IMPORTANT: @warn "isterminal FIX" #f17726
    end
    if mdp.sim_hash != s.hash
        go_to_state(mdp, s) # TODO: fix! Where should this happen? in action_info/select_action
    end
    return BlackBox.isterminal(mdp.sim)
end


POMDPs.discount(mdp::ASTMDP) = 1.0 # Undiscounted


# TODO: POMDPs.action # i.e. random_action
# This called by the rollout!
function POMDPs.action(policy::RandomPolicy, s::ASTState)
    mdp = policy.problem
    rsg::RSG = mdp.rsg
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
    # return [random_action(mdp) for k in 1:10]
    return [random_action(mdp)]
end


function go_to_state(mdp::ASTMDP, target_state::ASTState)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    BlackBox.initialize(mdp.sim)
    actions = get_action_sequence(target_state)
    R = 0.0
    for a in actions
        s, r = gen(mdp, s, a, Random.GLOBAL_RNG)
        R += r
    end
    @assert s == target_state
    return (R, actions)
end

global DDD = true # TODO: Remove

function rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        global DDD
        if DDD
            @show s
            DDD = false
        end

        # TODO: encapsulate this.
        # Save top k paths from rollout

        # TODO: Put this into one call (save loops)
        ancestor_actions = action_trace(s)
        q_value = q_trace(s)

        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff11
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff22
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff33
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff44
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff55
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff66
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff77
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff88
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff99
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #101110
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #102211
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #103312
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #104413
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #105514
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #106615
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #107716
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #108817
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #109918
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #100019
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #101120
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #102221
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #103322
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #104423
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #105524
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #106625
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #107726
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff11
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff22
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff33
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff44
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff55
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff66
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff77
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff88
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #10ff99
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #101110
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #102211
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #103312
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #104413
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #105514
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #106615
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #107716
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #108817
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #109918
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #100019
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #101120
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #102221
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #103322
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #104423
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #105524
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #106625
        # TODO IMPORTANT: Record q_values from non-rollout (i.e. from MCTS) #107726


        top_k = 20
        if !haskey(mdp.top_paths, s.hash)
            enqueue!(mdp.top_paths, ancestor_actions, q_value)
            while length(mdp.top_paths) > top_k
                dequeue!(mdp.top_paths)
            end
        end

        # go_to_state(mdp, s)

        return 0.0
    else
        a::ASTAction = random_action(mdp) # TODO: Use "POMDPs.action", requires MCTS planner

        # push_action!(mdp.tracker, a) # TODO: Remove

        (sp, r) = gen(DDNOut(:sp, :r), mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, d-1)

        # sp.q_value = q_value
        # push_q_value_rev!(mdp.tracker, q_value) # TODO: Remove
        return q_value
    end
end



function get_action_sequence(s::ASTState)
    actions::Vector{ASTAction} = ASTAction[]

    # Trace up the tree
    while !isnothing(s.parent)
        prepend!(actions, [s.action])
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
# function next_action(mdp::ASTMDP, s::ASTState, snode)
function select_action(mdp::ASTMDP, s::ASTState, snode) # analogous to "action_info"
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

        # This calls MCTS.simulate!

        hr = HistoryRecorder()
        history = simulate(hr, mdp, policy)
        # @show history
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
        end

        # TODO: cpu timing cut-off
    end

    # TODO: track actions and q_values internal to MCTS ?

    # Return simulation to current state
    go_to_state(mdp, s)

    # Extract actions taken at current state
    display(snode)
    # q = snode.tree.q[snode.tree.s_lookup[s.parent]]

    # a = action(planner, s) # TODO: Might cause StackOverflow...

    return random_action(mdp)
    # return a # TODO: return best action
end


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
Return optimal action path from MCTS tree (using `info[:tree]` from `(, info) = action_info(...)`).
"""
function get_optimal_path(mdp, tree, snode::Int, actions::Vector)
    best_Q = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end

    print("State = 0x", string(tree.s_labels[snode].hash, base=16), "\t:\t")
    if sanode != 0
        print("Q = ", tree.q[sanode], "\t:\t")
        println("Action = ", tree.a_labels[sanode].rsg.state)
        push!(actions, tree.a_labels[sanode])

        # Find subsequent maximizing state node
        best_Q_a = -Inf
        snode2 = 0
        for tran in tree.transitions[sanode]
            if tran[2] > best_Q_a
                best_Q_a = tran[2]
                snode2 = tran[1]
            end
        end

        if snode2 != 0
            get_optimal_path(mdp, tree, snode2, actions)
        end
    else
        AST.go_to_state(mdp, tree.s_labels[snode])
        if BlackBox.isevent(mdp.sim)
            println("Event.")
        else
            println("End of tree.")
        end
    end

    return actions
end
get_optimal_path(mdp, tree, state, actions=[]) = get_optimal_path(mdp, tree, tree.s_lookup[state], actions)



function playback(mdp::ASTMDP, actions)
    # k = collect(keys(mdp.top_paths)) # Used for history... TODO: remove
    # actions = k[end]

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


"""
Follow MCTS optimal path online calling `action` after each selected state.
"""
function online_path(mdp::MDP, planner::Policy)
    # Follow MCTS policy online.
    s = initialstate(mdp, Random.GLOBAL_RNG)
    a = action(planner, s)
    BlackBox.initialize(mdp.sim)

    actions = ASTAction[a]

    while true
        println("Sim. state: ", mdp.sim.x, " -> ", "Action: ", a.rsg.state)

        if BlackBox.isterminal(mdp.sim)
            break
        else
            a = action(planner, s)
            push!(actions, a)
            go_to_state(mdp, s)
            (sp, r) = gen(mdp, s, a, Random.GLOBAL_RNG)
            s = sp
        end
    end

    return actions
end



"""
Trace up the tree to get all ancestor states.
"""
function state_trace(s::ASTState)
    states::Vector{ASTState} = [s]
    while !isnothing(s.parent)
        prepend!(states, [s.parent])
        s = s.parent
    end
    return states::Vector{ASTState}
end


"""
Trace up the tree to get all ancestor actions.
"""
function action_trace(s::ASTState)
    actions = [s.action]
    while !isnothing(s.parent)
        prepend!(actions, [s.parent.action])
        s = s.parent
    end
    return actions
end

"""
Trace up the tree and accumulate Q-values.
"""
function q_trace(s::ASTState)
    q_value::Float64 = s.q_value
    while !isnothing(s.parent)
        q_value += s.parent.q_value
        s = s.parent
    end
    return q_value::Float64
end


end  # module AST