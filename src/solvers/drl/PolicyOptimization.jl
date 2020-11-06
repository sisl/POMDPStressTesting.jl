module PolicyOptimization

using ..AST
using Flux
using Flux.Optimise: update!
using Zygote
using POMDPs
using Random
using Parameters
using RLInterface
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
       PPOPlanner

# Modified from Shreyas Kowshik's implementation.
include("policies.jl")
include("trpo.jl")
include("ppo.jl")

include(joinpath("utils", "utils.jl"))
include(joinpath("utils", "buffer.jl"))
include(joinpath("utils", "policy_saving.jl"))

include("rollout.jl")
include("train.jl")


function POMDPs.action(planner::Union{TRPOPlanner, PPOPlanner}, s)
    if actiontype(planner.mdp) == ASTSeedAction
        @warn "DRL solvers (TRPO/PPO) are not as effective with ASTSeedAction. Use ASTMDP{ASTSampleAction}() instead."
    end
    train!(planner) # train neural networks

    # Pass back action trace if recording is on (i.e. top_k)
    if planner.mdp.params.top_k > 0
        return get_top_path(planner.mdp)
    else
        statevec = convert_s(Vector{Float32}, s, planner.mdp)
        nn_action = get_action(planner.policy, statevec)
        ast_action = translate_ast_action(planner.mdp.sim, nn_action, actiontype(planner.mdp))
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
