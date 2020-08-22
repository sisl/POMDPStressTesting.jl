"""
Provides virtual interface for Adaptive Stress Testing (AST) formulation of MDPs/POMDPs
"""
module AST

using Random
using Parameters
using DataStructures
using POMDPPolicies
using POMDPSimulators
using POMDPs
using MCTS

export
    BlackBox,

    ASTParams,
    ASTMDP,
    ASTState,
    ASTAction,
    ASTMetrics,

    reward,
    action,
    initialstate,
    isterminal,

    playback,
    playout,
    get_top_path,
    online_path,

    MCTSASTSolver


include("BlackBox.jl")
include("ast_types.jl")
include("seeding.jl")
include("hashing.jl")
include("recording.jl")
include(joinpath("solvers", "mcts.jl"))




"""
    reward(s::State)::Float64

Reward function for the AST formulation. Defaults to:

    R_E       if isterminal and isevent (1)
    -d        if isterminal and !isevent (2)
    log(p)    otherwise (3)

    1) Terminates with event, collect reward_bonus (defaults to 0)
    2) Terminates without event, collect negative miss distance
    3) Each non-terminal step, accumulate reward correlated with the transition probability


For epsidic reward problems (i.e. rewards only at the end of an episode), set `mdp.episodic_rewards` to get:
    (log(p) - d)*R_E    if isterminal and isevent (1)
    log(p) - d          if isterminal and !isevent (2)
    0                   otherwise (3)

    1) Terminates with event, collect transition probability and miss distance with multiplicative reward bonus
    2) Terminates without event, collect transitions probability and miss distance
    3) Each non-terminal step, no intermediate reward (set `mdp.give_intermediate_reward` to use log transition probability)
"""
function POMDPs.reward(mdp::ASTMDP, logprob::Float64, isevent::Bool, isterminal::Bool, miss_distance::Float64)
    if logprob > 0
        error("Make sure BlackBox.transition! outputs the log-probability.")
    end

    if mdp.episodic_rewards
        r = 0
        if isterminal
            r += logprob - miss_distance
            if isevent
                r *= mdp.reward_bonus # R_E (multiplicative)
            end
        else
            intermediate_reward = mdp.give_intermediate_reward ? logprob : 0
            r += intermediate_reward
        end
    else # Standard AST reward function
        r = logprob
        if isevent
            r += mdp.reward_bonus # R_E (additive)
        elseif isterminal
            r += -miss_distance # Only add miss distance cost if is terminal and not an event.
        end
    end

    if !mdp.episodic_rewards || (mdp.episodic_rewards && isterminal) || (mdp.episodic_rewards && mdp.give_intermediate_reward)
        record(mdp, prob=exp(logprob), logprob=logprob, miss_distance=miss_distance, reward=r)
    end

    return r
end



"""
Initialize AST MDP state. Overridden from `POMDPs.initialstate` interface.
"""
function POMDPs.initialstate(mdp::ASTMDP, rng::AbstractRNG=Random.GLOBAL_RNG) # rng unused.
    mdp.t_index = 1
    BlackBox.initialize!(mdp.sim)

    if !isnothing(mdp.params.reset_seed)
        mdp.params.seed = mdp.params.reset_seed
    end

    s::ASTState = ASTState(mdp.t_index, ASTAction(mdp.params.seed))
    mdp.sim_hash = s.hash
    return s::ASTState
end



"""
Generate next state and reward for AST MDP (handles episodic reward problems). Overridden from `POMDPs.gen` interface.
"""
function POMDPs.gen(mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG)
    @assert mdp.sim_hash == s.hash
    mdp.t_index += 1
    set_global_seed(a)
    hasproperty(mdp.sim, :actions) ? push!(mdp.sim.actions, a) : nothing

    # Step black-box simulation
    if mdp.episodic_rewards
        # Do not evaluate when problem has episodic rewards
        logprob::Float64 = BlackBox.transition!(mdp.sim)
        isevent::Bool = false
        miss_distance::Float64 = NaN
    else
        (logprob, miss_distance, isevent) = BlackBox.evaluate!(mdp.sim)
    end

    # Update state
    sp = ASTState(mdp.t_index, s, a)
    mdp.sim_hash = sp.hash
    sp.terminal = mdp.episodic_rewards ? false : BlackBox.isterminal!(mdp.sim) # termination handled by end-of-rollout
    r::Float64 = reward(mdp, logprob, isevent, sp.terminal, miss_distance)
    sp.q_value = r

    return (sp=sp, r=r)
end



"""
Determine if AST MDP is in a terminal state. Overridden from `POMDPs.isterminal` interface.
"""
function POMDPs.isterminal(mdp::ASTMDP, s::ASTState)
    @assert mdp.sim_hash == s.hash
    return BlackBox.isterminal!(mdp.sim)
end



"""
AST problems are (generally) undiscounted to treat future reward equally. Overridden from `POMDPs.discount` interface.
"""
POMDPs.discount(mdp::ASTMDP) = mdp.discount



"""
Randomly select next action, independent of the state.
"""
function random_action(mdp::ASTMDP)
    # Note, cannot use `rand` as that's controlled by the global seed.
    set_seed!(mdp)
    return ASTAction(mdp.current_seed)
end
random_action(mdp::ASTMDP, ::ASTState) = random_action(mdp)



"""
Randomly select next action, independent of the state. Overridden from `POMDPs.action` interface.
"""
POMDPs.action(policy::RandomPolicy, s::ASTState) = random_action(policy.problem)
POMDPs.actions(mdp::ASTMDP) = [random_action(mdp)]



"""
Reset AST simulation to a given state; used by the MCTS DPWSolver as the `reset_callback` function.
"""
function go_to_state(mdp::ASTMDP, target_state::ASTState)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    BlackBox.initialize!(mdp.sim)
    actions = action_trace(target_state)
    R = 0.0
    for a in actions
        s, r = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        R += r
    end
    @assert s == target_state

    record_trace(mdp, actions, R)

    return (R, actions)
end



"""
Record the best paths from termination leaf node.
"""
function record_trace(mdp::ASTMDP, actions::Vector{ASTAction}, reward::Float64)
    if mdp.params.top_k > 0 && BlackBox.isterminal!(mdp.sim)
        if !haskey(mdp.top_paths, actions)
            enqueue!(mdp.top_paths, actions, reward)
            while length(mdp.top_paths) > mdp.params.top_k
                dequeue!(mdp.top_paths)
            end
        end
    end
end



"""
Get k-th top path from the recorded `top_paths`.
"""
get_top_path(mdp::ASTMDP, k=mdp.params.top_k) = collect(keys(mdp.top_paths))[k]



"""
Rollout simulation for MCTS; used by the MCTS DPWSolver as the `estimate_value` function.
Custom rollout records action trace once the depth has been reached.
"""
function rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        go_to_state(mdp, s) # Records trace through this call
        return 0.0
    else
        a::ASTAction = random_action(mdp)

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*rollout(mdp, sp, d-1)

        return q_value
    end
end



"""
Rollout to only execute SUT at end (`p` accounts for probabilities generated outside the rollout)

User defined:
`feed_gen` Function to feed best action, replaces call to `gen` when feeding
`feed_type` Indicate when to feed best action. Either at the start of the rollout `:start`, or mid-rollout `:mid`
`best_callback` Callback function to record best miss distance or reward for later feeding during rollout
"""
function rollout_end(mdp::ASTMDP, s::ASTState, d::Int64; max_depth=-1, feed_gen=missing, feed_type::Symbol=:none, best_callback::Function=(sm,r)->sm)
    sim = mdp.sim
    a::ASTAction = random_action(mdp)

    if d == 0 || isterminal(mdp, s)
        # End-of-rollout evaluations.

        set_global_seed(a)
        hasproperty(sim, :actions) ? push!(sim.actions, a) : nothing

        # Step black-box simulation
        (prob::Float64, miss_distance::Float64, isevent::Bool) = BlackBox.evaluate!(sim)

        # Update state
        sp = ASTState(mdp.t_index, s, a)
        mdp.sim_hash = sp.hash
        sp.terminal = BlackBox.isterminal!(sim)
        r::Float64 = reward(mdp, prob, isevent, sp.terminal, miss_distance)
        sp.q_value = r

        best_callback(sim, miss_distance) # could use `r` instead

        return r
    else
        # Start of rollout best-action feeding OR mid-rollout best-action feeding.
        feed_available::Bool = !ismissing(feed_gen) && feed_type != :none
        start_of_rollout_feed::Bool = feed_type == :start && d == max_depth-1
        mid_rollout_feed::Bool = feed_type == :mid && d == div(max_depth,2)

        if feed_available && (start_of_rollout_feed || mid_rollout_feed)
            (sp, r) = feed_gen(mdp, s, a, Random.GLOBAL_RNG)
        else
            (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        end

        # Note, pass all keywords.
        q_value = r + discount(mdp)*rollout_end(mdp, sp, d-1; max_depth=max_depth, feed_gen=feed_gen, feed_type=feed_type, best_callback=best_callback)

        return q_value
    end
end



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
            println("Action = ", string(tree.a_labels[sanode]))
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
            if BlackBox.isevent!(mdp.sim)
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
function playback(mdp::ASTMDP, actions::Vector{ASTAction}, func=nothing; verbose=true)
    rng = Random.GLOBAL_RNG # Not used.
    s = initialstate(mdp, rng)
    BlackBox.initialize!(mdp.sim)
    display_trace::Bool = verbose && !isnothing(func)
    display_trace ? @show(func(mdp.sim)) : nothing

    for a in actions
        (sp, r) = @gen(:sp, :r)(mdp, s, a, rng)
        s = sp
        display_trace ? @show(func(mdp.sim)) : nothing
    end

    return s::ASTState # Returns final state
end



"""
Follow MCTS optimal path online calling `action` after each selected state.
"""
function online_path(mdp::MDP, policy::Policy, printstep=(sim, a)->println("Action: ", string(a));
                     verbose::Bool=false)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    BlackBox.initialize!(mdp.sim)

    # First step
    a = action(policy, s)
    actions = ASTAction[a]
    verbose ? printstep(mdp.sim, a) : nothing
    (s, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)

    while !BlackBox.isterminal!(mdp.sim)
        a = action(policy, s)
        push!(actions, a)
        verbose ? printstep(mdp.sim, a) : nothing
        (s, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
    end

    # Last step: last action is NULL.
    verbose ? printstep(mdp.sim, ASTAction()) : nothing

    if BlackBox.isevent!(mdp.sim)
        @info "Event found!"
    else
        @info "Hit terminal state."
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
    actions::Vector{ASTAction} = ASTAction[]

    # Trace up the tree
    while !isnothing(s.parent)
        prepend!(actions, [s.action])
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
    Play out the optimal path given an MDP and a policy.

    This is the main entry function to get a full failure trajectory from the policy.
"""
function playout(mdp, policy::DPWPlanner; return_tree::Bool=false)
    initstate = initialstate(mdp)
    tree = MCTS.action_info(policy, initstate, tree_in_info=true, show_progress=true)[2][:tree] # this runs MCTS
    action_path::Vector = get_optimal_path(mdp, tree, initstate, verbose=true)

    return return_tree ? tree : action_path
end



end  # module AST