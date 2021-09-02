"""
Provides implementation of Adaptive Stress Testing (AST) formulation of MDPs/POMDPs.
"""
module AST

using Random
using CommonRLInterface
using Parameters
using Distributions
using DataStructures
using POMDPPolicies
using POMDPSimulators
using POMDPs

export
    BlackBox,
    GrayBox,

    ASTParams,
    ASTMDP,
    ASTState,
    ASTAction,
    ASTSeedAction,
    ASTSampleAction,
    ASTMetrics,
    actiontype,

    reward,
    action,
    initialstate,
    isterminal,
    hash_uint32,
    reset_metrics!,
    most_likely_failure,
    combine_ast_metrics,

    search!,
    playback,
    get_top_path,
    logpdf,
    sample,
    online_path

include("BlackBox.jl")
include("GrayBox.jl")
include("ast_types.jl")
include(joinpath("utils", "rand.jl"))
include(joinpath("utils", "seeding.jl"))
include(joinpath("utils", "hashing.jl"))
include(joinpath("utils", "recording.jl"))

# Julia 1.1 compat
if !@isdefined(hasproperty)
    hasproperty(x, s::Symbol) = s in propertynames(x)
end

# Julia 1.0 compat
if !@isdefined(isnothing)
    isnothing(x) = x === nothing
end



"""
    reward(mdp::ASTMDP, logprob::Real, isevent::Bool, isterminal::Bool, miss_distance::Real)::Float64

Reward function for the AST formulation. Defaults to:

    R_E       if isterminal and isevent (1)
    -d        if isterminal and !isevent (2)
    log(p)    otherwise (3)

    1) Terminates with event, collect reward_bonus (defaults to 0)
    2) Terminates without event, collect negative miss distance
    3) Each non-terminal step, accumulate reward correlated with the transition probability


For epsidic reward problems (i.e. rewards only at the end of an episode), set `mdp.params.episodic_rewards` to get:

    (log(p) - d)*R_E    if isterminal and isevent (1)
    log(p) - d          if isterminal and !isevent (2)
    0                   otherwise (3)

    1) Terminates with event, collect transition probability and miss distance with multiplicative reward bonus
    2) Terminates without event, collect transitions probability and miss distance
    3) Each non-terminal step, no intermediate reward (set `mdp.params.give_intermediate_reward` to use log transition probability)
"""
function POMDPs.reward(mdp::ASTMDP, logprob::Real, isevent::Bool, isterminal::Bool, miss_distance::Real, rate::Real)
    # Allows us to add keyword arguments like `record_metrics`
    return local_reward(mdp, logprob, isevent, isterminal, miss_distance, rate; record_metrics=true)
end

"""
Called by `reward` from POMDPs but allows us to add keyword arguments like `record_metrics`.
"""
function local_reward(mdp::ASTMDP, logprob::Real, isevent::Bool, isterminal::Bool, miss_distance::Real, rate::Real; record_metrics::Bool=true)
    if mdp.params.episodic_rewards
        r = 0
        if isterminal
            r += logprob - miss_distance
            if isevent
                r *= mdp.params.reward_bonus # R_E (multiplicative)
            end
        else
            intermediate_reward = mdp.params.give_intermediate_reward ? logprob : 0
            r += intermediate_reward
        end
    else # Standard AST reward function
        # always apply logprob to capture all likelihoods
        r = logprob
        if isterminal && isevent
            r += mdp.params.reward_bonus # R_E (additive)
        elseif isterminal && !isevent
            r += -miss_distance # Only add miss distance cost if is terminal and not an event.
        end
        if mdp.params.use_potential_based_shaping
            r += rate # potential-based reward shaping
        end
    end

    if !isnothing(mdp.predict)
        # Reward shape based on failure prediction.
        r += mdp.predict([rate, miss_distance])
    end

    if record_metrics
        if !mdp.params.episodic_rewards || (mdp.params.episodic_rewards && isterminal) || (mdp.params.episodic_rewards && mdp.params.give_intermediate_reward)
            record(mdp, prob=exp(logprob), logprob=logprob, miss_distance=miss_distance, reward=r, event=isevent, terminal=isterminal, rate=rate)
        end
    end

    if isterminal
        # end of episode
        record_returns(mdp)
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

    if actiontype(mdp) == ASTSeedAction
        s = ASTState(t_index=mdp.t_index, action=ASTSeedAction(mdp.params.seed))
    elseif actiontype(mdp) == ASTSampleAction
        s = ASTState(t_index=mdp.t_index)
    end

    mdp.sim_hash = s.hash
    return s::ASTState
end



"""
Generate next state and reward for AST MDP (handles episodic reward problems). Overridden from `POMDPs.gen` interface.
"""
function POMDPs.gen(mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG=Random.GLOBAL_RNG)
    # Allows us to add keyword arguments like `record`
    return local_gen(mdp, s, a, rng; record=true)
end

"""
Called by @gen from POMDPs but allows us to add keyword arguments like `record`.
"""
function local_gen(mdp::ASTMDP, s::ASTState, a::ASTAction, rng::AbstractRNG=Random.GLOBAL_RNG; record::Bool=true)
    @assert mdp.sim_hash == s.hash
    if mdp.t_index == 1 # initial state indication
        prev_distance = 0
    else
        prev_distance = BlackBox.distance(mdp.sim)
    end
    mdp.t_index += 1
    isa(a, ASTSeedAction) ? set_global_seed(a) : nothing
    hasproperty(mdp.sim, :actions) ? push!(mdp.sim.actions, a) : nothing

    # Step black-box simulation
    if mdp.params.episodic_rewards
        # Do not evaluate when problem has episodic rewards
        if a isa ASTSeedAction
            if mdp.params.pass_seed_action
                logprob = GrayBox.transition!(mdp.sim, a.seed)
            else
                logprob = GrayBox.transition!(mdp.sim)
            end
        elseif a isa ASTSampleAction
            logprob = GrayBox.transition!(mdp.sim, a.sample)
        end
        isevent::Bool = false
        miss_distance::Float64 = NaN
        rate::Float64 = NaN
    else
        if a isa ASTSeedAction
            if mdp.params.pass_seed_action
                (logprob, miss_distance, isevent) = BlackBox.evaluate!(mdp.sim, a.seed)
            else
                (logprob, miss_distance, isevent) = BlackBox.evaluate!(mdp.sim)
            end
        elseif a isa ASTSampleAction
            (logprob, miss_distance, isevent) = BlackBox.evaluate!(mdp.sim, a.sample)
        end
        rate = BlackBox.rate(prev_distance, mdp.sim)
    end

    # Update state
    sp = ASTState(t_index=mdp.t_index, parent=s, action=a)
    mdp.sim_hash = sp.hash
    mdp.rate = rate
    sp.terminal = mdp.params.episodic_rewards ? false : BlackBox.isterminal(mdp.sim) # termination handled by end-of-rollout
    r::Float64 = local_reward(mdp, logprob, isevent, sp.terminal, miss_distance, rate; record_metrics=record)
    sp.q_value = r

    # TODO: Optimize (debug?)
    if record
        record_trace(mdp, action_q_trace(sp)...)
    end

    return (sp=sp, r=r)
end



"""
Determine if AST MDP is in a terminal state. Overridden from `POMDPs.isterminal` interface.
"""
function POMDPs.isterminal(mdp::ASTMDP, s::ASTState)
    @assert mdp.sim_hash == s.hash
    return BlackBox.isterminal(mdp.sim)
end



"""
AST problems are (generally) undiscounted to treat future reward equally. Overridden from `POMDPs.discount` interface.
"""
POMDPs.discount(mdp::ASTMDP) = mdp.params.discount



"""
Randomly select next action, independent of the state.
"""
random_action(mdp::ASTMDP, ::ASTState) = random_action(mdp)
function random_action(mdp::ASTMDP{ASTSeedAction})
    # Note, cannot use `rand` as that's controlled by the global seed.
    set_seed!(mdp)
    return ASTSeedAction(mdp.current_seed)
end

function random_action(mdp::ASTMDP{ASTSampleAction})
    environment::GrayBox.Environment = GrayBox.environment(mdp.sim)
    sample::GrayBox.EnvironmentSample = rand(environment)
    return ASTSampleAction(sample)
end



"""
Randomly select next action, independent of the state. Overridden from `POMDPs.action` interface.
"""
POMDPs.action(policy::RandomPolicy, s::ASTState) = random_action(policy.problem)
"""
Randomly select next actions, independent of the state. Overridden from `POMDPs.actions` interface.
"""
POMDPs.actions(mdp::ASTMDP) = [random_action(mdp)]



import Base.rand
rand(rng::AbstractRNG, s::ASTState) = s



"""
Reset AST simulation to a given state; used by the MCTS DPWSolver as the `reset_callback` function.
"""
function go_to_state(mdp::ASTMDP, target_state::ASTState; record=true)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    BlackBox.initialize!(mdp.sim)
    actions = action_trace(target_state)
    R = 0.0
    for a in actions
        s, r = local_gen(mdp, s, a, Random.GLOBAL_RNG; record=record)
        R += r
    end
    @assert s == target_state

    return (R, actions)
end



global TMP_DATA_RATE = []
global TMP_DATA_DISTANCE = []

"""
Record the best paths from termination leaf node.
"""
function record_trace(mdp::ASTMDP, actions::Vector{ASTAction}, reward::Float64)
    if mdp.params.top_k > 0 && BlackBox.isterminal(mdp.sim)
        if !haskey(mdp.top_paths, actions)
            enqueue!(mdp.top_paths, actions, reward)
            while length(mdp.top_paths) > mdp.params.top_k
                dequeue!(mdp.top_paths)
            end
        end
    end

    # Data collection of {(𝐱=rates, y=isevent), ...}
    if mdp.params.collect_data
        global TMP_DATA_RATE, TMP_DATA_DISTANCE
        closure_rate = mdp.rate
        distance = BlackBox.distance(mdp.sim)
        push!(TMP_DATA_RATE, closure_rate)
        push!(TMP_DATA_DISTANCE, distance)

        if BlackBox.isterminal(mdp.sim)
            𝐱 = [actions, TMP_DATA_DISTANCE, TMP_DATA_RATE]
            y = BlackBox.isevent(mdp.sim)
            push!(mdp.dataset, (𝐱, y))
            TMP_DATA_DISTANCE = []
            TMP_DATA_RATE = []
        end
    end

    # # Data collection of {(𝐱=disturbances, y=isevent), ...}
    # if mdp.params.collect_data && BlackBox.isterminal(mdp.sim)
    #     closure_rate = mdp.rate
    #     distance = BlackBox.distance(mdp.sim)
    #     𝐱 = vcat(actions, distance, closure_rate)
    #     y = BlackBox.isevent(mdp.sim)
    #     push!(mdp.dataset, (𝐱, y))
    # end
end



"""
Get k-th top path from the recorded `top_paths`.
"""
get_top_path(mdp::ASTMDP, k=min(mdp.params.top_k, length(mdp.top_paths))) = collect(keys(mdp.top_paths))[k]


"""
Return the action trace (i.e., trajectory) with the highest log-likelihood that led to failure.
"""
most_likely_failure(planner) = most_likely_failure(planner.mdp.metrics, planner.mdp.dataset)
function most_likely_failure(metrics::ASTMetrics, 𝒟)
    # TODO: get this index from the returned `action_trace` itself
    failure_trace = []
    if any(metrics.event)
        # Failures were found.
        event_indices = []
        event_logprob = []
        for (episode, i) in enumerate(findall(metrics.terminal))
            if metrics.event[i]
                push!(event_indices, episode)
                push!(event_logprob, metrics.logprob[i])
            end
        end
        most_likely_failure_episode_index = event_indices[argmax(event_logprob)]
        failure_trace = 𝒟[most_likely_failure_episode_index][1][1:end-2][1]
    end
    return convert(Vector{ASTAction}, failure_trace)
end


"""
Rollout simulation for MCTS; used by the MCTS DPWSolver as the `estimate_value` function.
Custom rollout records action trace once the depth has been reached.
"""
function rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    if d == 0 || isterminal(mdp, s)
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

        isa(a, ASTSeedAction) ? set_global_seed(a) : nothing
        hasproperty(sim, :actions) ? push!(sim.actions, a) : nothing
        rate = BlackBox.distance(sim)

        # Step black-box simulation
        if a isa ASTSeedAction
            if mdp.params.pass_seed_action
                (prob::Float64, miss_distance::Float64, isevent::Bool) = BlackBox.evaluate!(sim, a.seed)
            else
                (prob, miss_distance, isevent) = BlackBox.evaluate!(sim)
            end
        elseif a isa ASTSampleAction
            (prob, miss_distance, isevent) = BlackBox.evaluate!(sim, a.sample)
        end

        # Update state
        sp = ASTState(t_index=mdp.t_index, parent=s, action=a)
        mdp.sim_hash = sp.hash
        sp.terminal = BlackBox.isterminal(sim)
        r::Float64 = reward(mdp, prob, isevent, sp.terminal, miss_distance, rate)
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
Play back a given action trace from the `initialstate` of the MDP.
"""
playback(mdp::ASTMDP, actions::Nothing, func=nothing; kwargs...) = @warn("Action trace is `nothing`, please set mdp.params.top_k > 0.")
playback(planner, actions::Vector{ASTAction}, func=nothing; kwargs...) = playback(planner.mdp, actions, func; kwargs...)
function playback(mdp::ASTMDP, actions::Vector{ASTAction}, func=nothing; verbose=true, return_trace::Bool=false)
    rng = Random.GLOBAL_RNG # Not used.
    s = initialstate(mdp, rng)
    BlackBox.initialize!(mdp.sim)
    trace = []
    function trace_and_show(func)
        if !isnothing(func)
            single_step = func(mdp.sim)
            push!(trace, single_step)
            verbose && println(single_step)
        end
    end
    display_trace::Bool = verbose && !isnothing(func)
    trace_and_show(func)

    for a in actions
        isnothing(func) && verbose ? println(string(a)) : nothing
        (sp, r) = local_gen(mdp, s, a, rng; record=false)
        s = sp
        trace_and_show(func)
    end

    if return_trace
        return trace::Vector # Returns trace output by `func`
    else
        return s::ASTState # Returns final state
    end
end



"""
Follow MCTS optimal path online calling `action` after each selected state.
"""
function online_path(mdp::MDP, planner::Policy, printstep=(sim, a)->println("Action: ", string(a));
                     verbose::Bool=false)
    s = initialstate(mdp, Random.GLOBAL_RNG)
    BlackBox.initialize!(mdp.sim)

    # First step
    a = action(planner, s)
    actions = ASTAction[a]
    verbose ? printstep(mdp.sim, a) : nothing
    (s, r) = local_gen(mdp, s, a, Random.GLOBAL_RNG; record=false)

    while !BlackBox.isterminal(mdp.sim)
        a = action(planner, s)
        push!(actions, a)
        verbose ? printstep(mdp.sim, a) : nothing
        (s, r) = local_gen(mdp, s, a, Random.GLOBAL_RNG; record=false)
    end

    # Last step: last action is NULL.
    ActionType::Type = actiontype(mdp)
    verbose ? printstep(mdp.sim, ActionType()) : nothing

    if BlackBox.isevent(mdp.sim)
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
Trace up the tree to get all ancestor actions and summed Q-values.
"""
function action_q_trace(s::ASTState)
    actions::Vector{ASTAction} = ASTAction[]
    q_value::Float64 = s.q_value

    # Trace up the tree
    while !isnothing(s.parent)
        prepend!(actions, [s.action])
        q_value += s.parent.q_value
        s = s.parent
    end

    return actions::Vector, q_value::Float64
end


"""
Sum up the likelihoods of the entire trajectory.
"""
function Distributions.logpdf(action_trace::Vector{ASTAction})
    return sum([action_trace[t].sample[k].logprob for k in keys(action_trace[1].sample) for t in 1:length(action_trace)])
end


"""
Sum up the likelihood of the entire trajectory.
"""
function sample(action_trace::Vector{ASTAction})
    # TODO: categorical with weight logprob.
    rand([action_trace[t].sample[k].sample for k in keys(action_trace[1].sample) for t in 1:length(action_trace)])
end


"""
Clear data stored in `mdp.metrics`.
"""
function reset_metrics!(mdp::ASTMDP)
    mdp.metrics = ASTMetrics()
end


"""
    search!(planner)

Search for failures given a `planner`. Implemented by each solver.
"""
function search! end


end  # module AST
