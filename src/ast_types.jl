const DEFAULT_SEED = 0


"""
    AST.ASTParams

Adaptive Stress Testing specific simulation parameters.
"""
@with_kw mutable struct ASTParams
    max_steps::Int64 = 0 # Maximum simulation time step (for runaways in simulation)
    seed::UInt32 = DEFAULT_SEED # Initial seed value
    reset_seed::Union{Nothing, Int64} = nothing # Reset to this seed value on initialize()
    top_k::Int64 = 0 # Number of top performing paths to save (defaults to 0, i.e. do not record)
    debug::Bool = false # Flag to indicate debugging mode (i.e. metrics collection, etc)
    episodic_rewards::Bool = false # decision making process with epsidic rewards
    give_intermediate_reward::Bool = false # give log-probability as reward during intermediate gen calls (used only if `episodic_rewards`)
    reward_bonus::Float64 = episodic_rewards ? 100 : 0 # reward received when event is found, multiplicative when using `episodic_rewards`
    use_potential_based_shaping::Bool = true # apply potential-based reward shaping to speed up learning
    pass_seed_action::Bool = false # pass the selected RNG seed to the GrayBox.transition! and BlackBox.evaluate! functions
    collect_data::Bool = false # flag to indicate supervised dataset collection of (𝐱=disturbances, y=isevent)
    discount::Float64 = 1.0 # discount factor (generally 1.0 to not discount later samples in the trajectory)
end
ASTParams(max_steps::Int64, seed::Int64) = ASTParams(max_steps=max_steps, seed=seed)
ASTParams(max_steps::Int64, seed::Int64, top_k::Int64, debug::Bool) = ASTParams(max_steps=max_steps, seed=seed, top_k=top_k, debug=debug)



"""
Abstract type for the AST action variants.
"""
abstract type ASTAction end

"""
    AST.ASTSeedAction

Random seed AST action.
"""
@with_kw mutable struct ASTSeedAction <: ASTAction
    seed::UInt32 = DEFAULT_SEED
end

import Base.string
string(a::ASTSeedAction) = "0x" * string(a.seed, base=16)


"""
    AST.ASTSampleAction

Random environment sample as the AST action.
"""
@with_kw mutable struct ASTSampleAction <: ASTAction
    sample::Union{GrayBox.EnvironmentSample, Nothing} = nothing
end

function string(a::ASTSampleAction)
    strings = ["$k => $(a.sample[k].value) ($(a.sample[k].logprob))" for k in keys(a.sample)]
    return join(strings, ',')
end



"""
    AST.ASTState

State of the AST MDP.
"""
@with_kw mutable struct ASTState
    t_index::Int64 = 0 # Confidence check that time corresponds
    parent::Union{Nothing,ASTState} = nothing # Parent state, `nothing` if root
    action::Union{Nothing,ASTAction} = nothing # Action taken from parent
    state::GrayBox.State = nothing # State of the gray-box simulation (not required/available for all problems)
    hash::UInt64 = hash(t_index, parent, action) # Hash simulation state to match with ASTState
    q_value::Float64 = 0.0 # Saved Q-value
    terminal::Bool = false # Indication of termination state
end



"""
    AST.ASTMetrics

Debugging metrics.
"""
@with_kw mutable struct ASTMetrics
    miss_distance = Real[]
    rate = Real[]
    logprob = Real[]
    prob = Real[]
    reward = Real[]
    intermediate_reward = Real[] # for computing returns at episode termination
    returns = Vector{Real}[]
    event = Bool[]
    terminal = Bool[]
end



"""
    AST.ASTMDP

Adaptive Stress Testing MDP problem formulation object.
"""
@with_kw mutable struct ASTMDP{Action<:ASTAction} <: MDP{ASTState, Action}
    params::ASTParams = ASTParams() # AST simulation parameters
    sim::GrayBox.Simulation # Black-box simulation struct
    sim_hash::UInt64 = hash(0) # Hash to keep simulations in sync
    t_index::Int64 = 1 # Simulation time
    current_seed::Union{Nothing, UInt32} = nothing # Rolling seed (`nothing` indicates initial seed comes from params.seed)

    top_paths::PriorityQueue{Any, Float64} = PriorityQueue{Any, Float64}(Base.Order.Forward) # Collection of best paths in the tree
    metrics::ASTMetrics = ASTMetrics() # Debugging metrics

    dataset::Vector = [] # (𝐱=disturbances, y=isevent) supervised dataset

    rate::Real = NaN # Stored current rate value

    predict::Union{Function, Nothing} = nothing # failure prediction function (negative = non-failures, positive = failures)
end

ASTMDP{Action}(sim::GrayBox.Simulation) where {Action<:ASTAction} = ASTMDP{Action}(sim=sim)
ASTMDP{Action}(params::ASTParams, sim::GrayBox.Simulation) where {Action<:ASTAction} = ASTMDP{Action}(params=params, sim=sim)

# Default to ASTSeedAction
ASTMDP(sim::GrayBox.Simulation) = ASTMDP{ASTSeedAction}(sim=sim)
ASTMDP(params::ASTParams, sim::GrayBox.Simulation) = ASTMDP{ASTSeedAction}(params=params, sim=sim)
