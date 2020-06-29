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
using MCTS

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
    # TODO: export necessary functions

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
    [✓] Specific AST struct???
        [✓] POMDP/MDP formulation of AST as ::MDP{S,A}
    [✓] Integrate with solvers (i.e. MCTS.jl with "single" progressive widening)
    [] @impl_dep: implementation dependencies (see pomdp.jl for example)
    [] Benchmarking/Results tools
    [] BlackBox.evaluate vs. BlackBox.step
=#

const DEFAULT_RSG_LENGTH = 3 # Default random seed generator (RSG) length (TODO)



# TODO: Generalize so be seeds or disturbance distributions directly.
mutable struct ASTAction
    rsg::RSG
    ASTAction(rsg::RSG) = new(rsg)
    ASTAction(len::Int64=DEFAULT_RSG_LENGTH, seed::Int64=0) = ASTAction(RSG(len, seed))
end


mutable struct ASTState
    t_index::Int64 # Sanity check that time corresponds
    hash::UInt64 # Hash simulation state to match with ASTState
    parent::Union{Nothing,ASTState} # Parent state, `nothing` if root
    action::ASTAction # Action taken from parent, 0 if root
    q_value::Float64 # Saved Q-value
    terminal::Bool
end

# TODO: Put into ASTState struct contructor
function ASTState(t_index::Int64, parent::Union{Nothing,ASTState}, action::ASTAction)
    s = ASTState(t_index, UInt64(0), parent, action, 0.0, false)
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


"""
    AST.Params

Adaptive Stress Testing specific simulation parameters.
"""
@with_kw mutable struct Params
    max_steps::Int64 = 0 # Maximum simulation time step (for runaways in simulation)
    rsg_length::Int64 = 3 # Dictates number of unique available random seeds ((????))
    init_seed::Int64 = 0 # Initial seed value
    reset_seed::Union{Nothing, Int64} = nothing # Reset to this seed value on initialize()
    top_k::Int64 = 0 # Number of top performing paths to save (defaults to 0, i.e. do not record)
    distance_reward::Bool = false # When enabled, use the -miss_distance when either an event or terminal state is reached
    debug::Bool = false # Flag to indicate debugging mode (i.e. metrics collection, etc)
end
Params(max_steps::Int64, rsg_length::Int64, init_seed::Int64) = Params(max_steps=max_steps, rsg_length=rsg_length, init_seed=init_seed)
Params(max_steps::Int64, rsg_length::Int64, init_seed::Int64, top_k::Int64, distance_reward::Bool, debug::Bool) = Params(max_steps=max_steps, rsg_length=rsg_length, init_seed=init_seed, top_k=top_k, distance_reward=distance_reward, debug=debug)



"""
    AST.ASTMetrics

Debugging metrics.
"""
@with_kw mutable struct ASTMetrics
    miss_distance = Any[]
    log_prob = Any[]
    prob = Any[]
    reward = Any[]
end



# TODO: Naming
"""
    AST.ASTMDP

Adaptive Stress Testing MDP simulation object.
"""
mutable struct ASTMDP <: MDP{ASTState, ASTAction}
    sequential::Bool # sequential decision making process or single/buffer-shot
    params::Params # AST simulation parameters
    sim::BlackBox.Simulation # Black-box simulation struct
    sim_hash::UInt64 # Hash to keep simulations in sync

    t_index::Int64 # TODO
    rsg::RSG # Random seed generator
    initial_rsg::RSG # Initial random seed generator
    reset_rsg::Union{Nothing,RSG} # Reset to this RSG if provided

    # TODO: {Tracker, Float64}...
    top_paths::PriorityQueue{Any, Float64} # Collection of best paths in the tree # TODO: Change 'Any'

    metrics::ASTMetrics # Debugging metrics

    function ASTMDP(params::Params, sim)
        rsg::RSG = RSG(params.rsg_length, params.init_seed)
        top_paths = PriorityQueue{Any, Float64}(Base.Order.Forward)
        return new(true, params, sim, hash(0), 1, rsg, deepcopy(rsg), !isnothing(params.reset_seed) ? RSG(params.rsg_length, params.reset_seed) : nothing, top_paths, ASTMetrics())
    end
end


# TODO: clear/push!
# TODO: put in separate file as Metrics Monitor?
"""
    AST.record(::ASTMDP, sym::Symbol, val)

Recard an ASTMetric specified by `sym`.
"""
function record(mdp::ASTMDP, sym::Symbol, val)
    if mdp.params.debug
        push!(getproperty(mdp.metrics, sym), val)
    end
end

function record(mdp::ASTMDP; prob=1, log_prob=exp(1), miss_distance=Inf, reward=-Inf)
    AST.record(mdp, :prob, prob)
    AST.record(mdp, :log_prob, log_prob)
    AST.record(mdp, :miss_distance, miss_distance)
    AST.record(mdp, :reward, reward)
end



"""
    reward(s::State)::Float64

Reward function for the AST formulation. Defaults to:

    0               s ∈ Event            # Terminates with event, maximum reward of 0
    -∞,             s ̸∈ Event and t ≥ T  # Terminates without event, maximum negative reward of -∞
    log P(s′ | s),  s ̸∈ Event and t < T  # Each non-terminal step, accumulate reward correlated with the transition probability
"""
function POMDPs.reward(mdp::ASTMDP, logprob::Float64, isevent::Bool, isterminal::Bool, miss_distance::Float64)
#=
    r = log(prob)
    if mdp.params.distance_reward
        # Alternate reward: use miss_distance at isevent to drive towards severity!
        if isevent || isterminal
            r += -miss_distance
        end
    else
        if isevent
            r += 0.0
        elseif isterminal
            # Add miss distance cost only if !isevent && isterminal
            r += -miss_distance
            # record(mdp, :prob, prob)
            # record(mdp, :log_prob, log(prob))
            # record(mdp, :miss_distance, miss_distance)
        end
    end
    # record(mdp, :reward, r)
=#

    # Eq. (11) alg4bb
    # if isevent || isterminal
    #     r = log(prob) # - miss_distance
    # else
    #     r = -miss_distance
    # end

    # TODO: Eq. (11) in alg4bb 
    # flipped
    # if isevent || isterminal

    # r = log(prob)
    # if isevent
    #     r += 0
    # elseif isterminal
    #     r += -miss_distance
    # end


    # TODO: consolodate or try different event bonuses.
    # r = log(prob)
#=
    r = logprob
    if isevent
        # r += 10*(-miss_distance)
        r += -miss_distance
    elseif isterminal
        r += -miss_distance
    end
=#
    r = logprob
    r += -miss_distance
    if isevent
        r *= 100
    end



    # r = -miss_distance
    # if isevent
    #     r *= 100
    # end



    # if isevent
    #     r = 10*(-miss_distance)
    # elseif isterminal
    #     r = -miss_distance
    #     # r = log(prob) - miss_distance
    # else
    #     r = log(prob)
    #     # r = -miss_distance
    # end

    # @show prob
    # record(mdp, :prob, prob)
    record(mdp, :prob, exp(logprob))
    # record(mdp, :log_prob, log(prob))
    record(mdp, :log_prob, logprob)
    record(mdp, :miss_distance, miss_distance)
    record(mdp, :reward, r)
    return r
end



# TODO: Handle `rng` (UNUSED)
"""
Initialize AST MDP state. Overridden from `POMDPs.initialstate` interface.
"""
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
# Generate next state and reward for AST
#=
"""
Generate next state and reward for AST MDP. Overridden from `POMDPs.gen` interface.
"""
function POMDPs.gen(mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG)
    @assert mdp.sim_hash == s.hash
    mdp.t_index += 1
    set_global_seed(a.rsg)

    # Step black-box simulation
    (prob::Float64, isevent::Bool, miss_distance::Float64) = BlackBox.evaluate(mdp.sim)

    # Update state
    sp = ASTState(mdp.t_index, s, a)
    mdp.sim_hash = sp.hash
    sp.terminal = BlackBox.isterminal(mdp.sim)
    r::Float64 = reward(mdp, prob, isevent, sp.terminal, miss_distance)
    sp.q_value = r

    return (sp=sp, r=r)
end
=#




# Generate next state and reward (0) for non-sequential problem
"""
TODO: Generate next state and reward for AST MDP. Overridden from `POMDPs.gen` interface.
"""
function POMDPs.gen(::DDNOut, mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG) # TODO. How to control `sequential`?
    @assert mdp.sim_hash == s.hash
    mdp.t_index += 1
    set_global_seed(a.rsg)

    # Step black-box simulation
    if mdp.sequential
        (prob::Float64, isevent::Bool, miss_distance::Float64) = BlackBox.evaluate(mdp.sim)
    else
        (prob, ) = BlackBox.transition_model(mdp.sim)
    end

    # Update state
    sp = ASTState(mdp.t_index, s, a)
    mdp.sim_hash = sp.hash
    sp.terminal = mdp.sequential ? BlackBox.isterminal(mdp.sim) : false
    r::Float64 = mdp.sequential ? reward(mdp, prob, isevent, sp.terminal, miss_distance) : 0
    sp.q_value = r

    return (sp=sp, r=r, p=prob)
end


"""
Determine if AST MDP is in a terminal state. Overridden from `POMDPs.isterminal` interface.
"""
function POMDPs.isterminal(mdp::ASTMDP, s::ASTState)
    @assert mdp.sim_hash == s.hash
    return BlackBox.isterminal(mdp.sim)
end


"""
AST problems are undiscounted to treat future reward equally AST MDP state. Overridden from `POMDPs.discount` interface.
"""
POMDPs.discount(mdp::ASTMDP) = 1.0


"""
Randomly select next action, independent of the state.
"""
function random_action(mdp::ASTMDP)
    rsg::RSG = mdp.rsg
    next!(rsg)
    return ASTAction(deepcopy(rsg))
end
random_action(mdp::ASTMDP, ::ASTState) = random_action(mdp)

"""
Randomly select next action, independent of the state. Overridden from `POMDPs.action` interface.
"""
POMDPs.action(policy::RandomPolicy, s::ASTState) = random_action(policy.problem)
POMDPs.actions(mdp::ASTMDP) = [random_action(mdp)] # TODO: Should this be handled better?



"""
Reset AST simulation to a given state; used by the MCTS DPWSolver as the `reset_callback` function.
"""
function go_to_state(mdp::ASTMDP, target_state::ASTState)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    BlackBox.initialize(mdp.sim)
    actions = get_action_sequence(target_state) # TODO: duplicate of action_trace
    R = 0.0
    P = 1
    for a in actions
        s, r, p = gen(DDNOut(:sp, :r, :p), mdp, s, a, Random.GLOBAL_RNG)
        R += r
        P *= p
    end
    @assert s == target_state

    record_trace(mdp, actions, R)

    return (R, actions, P)
end



"""
Record paths from leaf node that lead to an event.
"""
function record_trace(mdp::ASTMDP, actions::Vector{ASTAction}, summed_q_values::Float64)
    if mdp.params.top_k > 0 && BlackBox.isevent(mdp.sim) # TODO. include toggle for isevent requirement or not.
        if !haskey(mdp.top_paths, actions)
            enqueue!(mdp.top_paths, actions, summed_q_values)
            while length(mdp.top_paths) > mdp.params.top_k
                dequeue!(mdp.top_paths)
            end
        end
    end
end



"""
Rollout simulation for MCTS; used by the MCTS DPWSolver as the `estimate_value` function.
Custom rollout records action trace once the depth has been reached.
"""
function rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        # TODO Efficiency: To make this more efficient, collect trace as rollout is called (instead of tracing back up the tree)
        # TODO Efficiency: Call record_trace directly with the `action_trace` and `q_trace` outputs.
        go_to_state(mdp, s) # Records trace through this call
        # TODO: check to call evaluate.
        return 0.0
    else
        a::ASTAction = random_action(mdp) # TODO: Use "POMDPs.action", requires MCTS planner

        (sp, r) = gen(DDNOut(:sp, :r), mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, d-1)

        return q_value
    end
end


"""
Rollout to only execute SUT at end (`p` accounts for probabilities generated outside the rollout)
"""
function rollout_end(mdp::ASTMDP, s::ASTState, d::Int64; dead_end::Bool=true, p=1)
    a::ASTAction = random_action(mdp) # TODO: Use "POMDPs.action", requires MCTS planner

    # DEAD_END_ROLLOUT = true # short-circuit

    if dead_end || d == 0 || isterminal(mdp, s)
        # Step black-box simulation
        (prob::Float64, isevent::Bool, miss_distance::Float64) = BlackBox.evaluate(mdp.sim)

        # Update state
        sp = ASTState(mdp.t_index, s, a)
        mdp.sim_hash = sp.hash
        sp.terminal = BlackBox.isterminal(mdp.sim)
        r::Float64 = reward(mdp, p*prob, isevent, sp.terminal, miss_distance)
        sp.q_value = r

        return r
    else
        # (sp, r, p′) = gen(DDNOut(:sp, :r, :p), mdp, s, a, Random.GLOBAL_RNG)
        # q_value = r + discount(mdp)*rollout_end(mdp, sp, d-1; p=(p*p′))
        (sp, r) = gen(DDNOut(:sp, :r), mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout_end(mdp, sp, d-1; dead_end=dead_end)

        return q_value
    end
end



# Duplicate of action_trace. # TODO
function get_action_sequence(s::ASTState)
    actions::Vector{ASTAction} = ASTAction[]

    # Trace up the tree
    while !isnothing(s.parent)
        prepend!(actions, [s.action])
        s = s.parent
    end

    return actions::Vector{ASTAction}
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
function get_optimal_path(mdp, tree, snode::Int, actions::Vector{ASTAction}; verbose::Bool=false)
    best_Q = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end

    if verbose
        print("State = 0x", string(tree.s_labels[snode].hash, base=16), "\t:\t")
    end
    if sanode != 0
        if verbose
            print("Q = ", tree.q[sanode], "\t:\t")
            println("Action = ", tree.a_labels[sanode].rsg.state)
        end
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
            get_optimal_path(mdp, tree, snode2, actions, verbose=verbose)
        end
    else
        go_to_state(mdp, tree.s_labels[snode])
        if verbose
            if BlackBox.isevent(mdp.sim)
                println("Event.")
            else
                println("End of tree.")
            end
        end
    end

    return actions::Vector{ASTAction}
end
get_optimal_path(mdp, tree, state, actions::Vector{ASTAction}=ASTAction[]; kwargs...) = get_optimal_path(mdp, tree, tree.s_lookup[state], actions; kwargs...)



"""
Play back a given action trace from the `initialstate` of the MDP.
"""
function playback(mdp::ASTMDP, actions::Vector{ASTAction}, func=sim->sim.x; verbose=true)
    rng = Random.GLOBAL_RNG # Not used.
    s = initialstate(mdp, rng)
    BlackBox.initialize(mdp.sim)
    # TODO: This is Walk1D specific!
    @show func(mdp.sim)
    for a in actions
        (sp, r) = gen(DDNOut(:sp, :r), mdp, s, a, rng)
        s = sp
        # TODO: This is Walk1D specific!
        if verbose
            @show func(mdp.sim)
        end
    end
    return s::ASTState # Returns final state
end



"""
Follow MCTS optimal path online calling `action` after each selected state.
"""
function online_path(mdp::MDP, planner::Policy; verbose::Bool=false)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    BlackBox.initialize(mdp.sim)

    # TODO: This is Walk1D specific (mdp.sim.x)!
    printstep(mdp, a) = verbose ? println("Sim. state: ", mdp.sim.x, " -> ", "Action: ", a.rsg.state) : nothing

    # First step
    a = action(planner, s)
    actions = ASTAction[a]
    printstep(mdp, a)
    (s, r) = gen(DDNOut(:sp, :r), mdp, s, a, Random.GLOBAL_RNG)

    while !BlackBox.isterminal(mdp.sim)
        a = action(planner, s)
        push!(actions, a)
        (s, r) = gen(DDNOut(:sp, :r), mdp, s, a, Random.GLOBAL_RNG)
        printstep(mdp, a)
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


"""
    Play out the optimal path given an MDP and a policy/planner.

    This is the main entry function to get a full failure trajectory from the policy.
"""
function playout(mdp, planner)
    initstate = initialstate(mdp)
    tree = MCTS.action_info(planner, initstate, tree_in_info=true)[2][:tree] # see TreeVisualizer.visualize
    action_path::Vector = get_optimal_path(mdp, tree, initstate, verbose=true)
    return action_path
end


end  # module AST