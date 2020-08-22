const DEFAULT_SEED = 0


"""
    AST.ASTParams

Adaptive Stress Testing specific simulation parameters.
"""
@with_kw mutable struct ASTParams
    max_steps::Int64 = 0 # Maximum simulation time step (for runaways in simulation)
    seed::Int64 = DEFAULT_SEED # Initial seed value
    reset_seed::Union{Nothing, Int64} = nothing # Reset to this seed value on initialize()
    top_k::Int64 = 0 # Number of top performing paths to save (defaults to 0, i.e. do not record)
    debug::Bool = false # Flag to indicate debugging mode (i.e. metrics collection, etc)
end
ASTParams(max_steps::Int64, seed::Int64) = ASTParams(max_steps=max_steps, seed=seed)
ASTParams(max_steps::Int64, seed::Int64, top_k::Int64, debug::Bool) = ASTParams(max_steps=max_steps, seed=seed, top_k=top_k, debug=debug)



"""
    AST.ASTAction

Random seed AST action.
"""
@with_kw mutable struct ASTAction
    seed::UInt32 = DEFAULT_SEED
end

import Base.string
string(a::ASTAction) = "0x" * string(a.seed, base=16)



"""
    AST.ASTState

State of the AST MDP.
"""
mutable struct ASTState
    t_index::Int64 # Confidence check that time corresponds
    hash::UInt64 # Hash simulation state to match with ASTState
    parent::Union{Nothing,ASTState} # Parent state, `nothing` if root
    action::ASTAction # Action taken from parent
    q_value::Float64 # Saved Q-value
    terminal::Bool # Indication of termination state
end

function ASTState(t_index::Int64, parent::Union{Nothing,ASTState}, action::ASTAction)
    s = ASTState(t_index, 0, parent, action, 0.0, false)
    s.hash = hash(s)
    return s
end

function ASTState(t_index::Int64, action::ASTAction)
    return ASTState(t_index, nothing, action)
end





"""
    AST.ASTMetrics

Debugging metrics.
"""
@with_kw mutable struct ASTMetrics
    miss_distance = Any[]
    logprob = Any[]
    prob = Any[]
    reward = Any[]
end



"""
    AST.ASTMDP

Adaptive Stress Testing MDP problem formulation object.
"""
@with_kw mutable struct ASTMDP <: MDP{ASTState, ASTAction}
    episodic_rewards::Bool = false # decision making process with epsidic rewards
    give_intermediate_reward::Bool = false # give log-probability as reward during intermediate gen calls (used only if `episodic_rewards`)
    discount::Float64 = 1.0
    reward_bonus::Float64 = episodic_rewards ? 100 : 0 # reward received when event is found, multiplicative when using `episodic_rewards`
    params::ASTParams = ASTParams() # AST simulation parameters
    sim::BlackBox.Simulation # Black-box simulation struct
    sim_hash::UInt64 = hash(0) # Hash to keep simulations in sync
    t_index::Int64 = 1 # Simulation time
    current_seed::UInt32 = ASTAction().seed

    top_paths::PriorityQueue{Any, Float64} = PriorityQueue{Any, Float64}(Base.Order.Forward) # Collection of best paths in the tree
    metrics::ASTMetrics = ASTMetrics() # Debugging metrics
end

ASTMDP(sim::BlackBox.Simulation) = ASTMDP(sim=sim)
ASTMDP(params::ASTParams, sim::BlackBox.Simulation) = ASTMDP(params=params, sim=sim)
