module PolicyOptimization

using ..AST
using Flux
using Flux.Optimise: update!
using Zygote
using POMDPs
using Random
using Parameters
using POMDPModelTools
using CommonRLInterface
using Distributed
using Distributions
using LinearAlgebra
using Base.Iterators
import JLD2: @save, @load
import ProgressMeter: Progress, next!

export search!,
       TRPOSolver,
       TRPOPlanner,
       PPOSolver,
       PPOPlanner,
       disconunted_returns,
       get_action,
       sample_action

# Modified from Shreyas Kowshik's implementation.
include("policies.jl")
include("trpo.jl")
include("ppo.jl")

include(joinpath("utils", "utils.jl"))
include(joinpath("utils", "buffer.jl"))
include(joinpath("utils", "policy_saving.jl"))

include("rollout.jl")
include("train.jl")


"""
Used by the CommonRLInterface to interact with deep RL solvers.
"""
function POMDPs.convert_s(::Type{T}, s::ASTState, mdp::ASTMDP) where T <: AbstractArray
    if isnothing(s.state)
        s.state = GrayBox.state(mdp.sim) # [s.hash]
    end
    return s.state
end


"""
Sample random action from DRL policy.
"""
function sample_action(planner::Union{TRPOPlanner, PPOPlanner}, statevec)
    nn_action = get_action(planner.policy, statevec)
    return translate_ast_action(planner.mdp.sim, nn_action, actiontype(planner.mdp))
end


function POMDPs.action(planner::Union{TRPOPlanner, PPOPlanner}, s)
    if isnothing(GrayBox.state(planner.mdp.sim))
        error("GrayBox.state(sim) is nothing, please define this function.")
    end
    if actiontype(planner.mdp) == ASTSeedAction
        @warn "DRL solvers (TRPO/PPO) are not as effective with ASTSeedAction. Use ASTMDP{ASTSampleAction}() instead."
    end
    train!(planner) # train neural networks

    # Pass back action trace if recording is on (i.e. top_k)
    if planner.mdp.params.top_k > 0
        return get_top_path(planner.mdp)
    else
        statevec = convert_s(Vector{Float32}, s, planner.mdp)
        ast_action = sample_action(planner, statevec)
        return ast_action
    end
end

"""
    AST.search!(planner::Union{TRPOPlanner, PPOPlanner})

Search using the `TRPOPlanner` or `PPOPlanner` from an initial AST state.
Pass back the best action trace.
"""
function AST.search!(planner::Union{TRPOPlanner, PPOPlanner})
    mdp::ASTMDP = planner.mdp
    Random.seed!(mdp.params.seed) # Determinism
    s = AST.initialstate(mdp)
    return action(planner, s)
end

end # module
