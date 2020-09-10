# Modified from Shreyas Kowshik's implementation.

@with_kw mutable struct PPOSolver
    episode_length::Int64 = 100 # Length of each episode
    terminate_horizon::Bool = false # Zero reward at end of horizon
    resume::Bool = false # Load saved policy and resume.

    # Optimization parameters
    η::Float64 = 1e-3 # learning rate η
    optimizer = ADAM(η) # Optimizer for the value function and policy neural network

    # Generalized advantage estimate (GAE) parameters
    gae_γ::Float64 = 0.99 # GAE-Gamma
    gae_λ::Float64 = 0.95 # GAE-Lambda

    # Training parameters
    num_episodes::Int64 = 10 # Number of epochs of interaction (equal to number of policy updates)
    ppo_epochs::Int64 = 10 # Number of steps of interaction (state-action pairs) for the agent and the environment in each episode
    batch_size::Int64 = episode_length # Size of the batches to perfom the update on
    cₚ::Float64 = 1.0 # policy loss coefficient
    cᵥ::Float64 = 1.0 # value loss coefficient
    cₑ::Float64 = 0.01 # entropy loss coefficient

    # PPO parameters
    ϵ::Float64 = 0.1 # PPO gradient clipping

    # Policy parameters
    hidden_layer_size::Int64 = 30 # Policy and value hidden layer size
    policy_type::Symbol = :continuous # :discrete or :continuous (TODO. infer)
    log_std::Float64 = log(1) # Initial log standard deviation used in the log-STD network

    # Environment parameters
    state_size::Int64 = 1 # Size of the state/observation space.
    action_size::Int64 = 1 # Size of the action space.

    # Frequencies and verbosity
    save::Bool = false
    save_frequency::Int64 = 10
    verbose_frequency::Int64 = 5
    verbose::Bool = false
    show_progress::Bool = true
end


mutable struct PPOPlanner{P<:Union{MDP,POMDP}}
    solver::PPOSolver
    mdp::P
    env::Union{MDPEnvironment,POMDPEnvironment}
    policy::Union{CategoricalPolicy,DiagonalGaussianPolicy,Nothing}
end


# Uses RLInterface environment wrapper
POMDPs.solve(solver::PPOSolver, mdp::MDP) = PPOPlanner(set_action_size!(solver, mdp), mdp, MDPEnvironment(mdp), nothing)
POMDPs.solve(solver::PPOSolver, mdp::POMDP) = PPOPlanner(set_action_size!(solver, mdp), mdp, POMDPEnvironment(mdp), nothing)


# NOTE: action and playback shared between TRPO and PPO (see PolicyOptimization.jl)


function loss(solver::PPOSolver, policy, states::Array, actions::Array, advantages::Array, returns::Array, old_log_probs::Array)
    new_log_probs = log_prob(policy, states, actions)

    # Surrogate loss computations
    ratio = exp.(new_log_probs .- old_log_probs)
    surr1 = ratio .* advantages
    surr2 = clamp.(ratio, (1.0 - solver.ϵ), (1.0 + solver.ϵ)) .* advantages
    policy_loss = mean(min.(surr1, surr2))

    value_predicted = policy.value_net(states)
    value_loss = mean((value_predicted .- returns).^2)

    entropy_loss = mean(entropy(policy, states))

    return -solver.cₚ*policy_loss + solver.cᵥ*value_loss - solver.cₑ*entropy_loss
end


function ppo_update!(solver::PPOSolver, policy, states::Array, actions::Array, advantages::Array, returns::Array, old_log_probs::Array, kl_vars)
    model_params = Flux.params(get_policy_params(policy)..., get_value_params(policy)...)

    # Calculate gradients
    gs = Tracker.gradient(() -> loss(solver, policy, states, actions, advantages, returns, old_log_probs), model_params)

    # Take a step of optimization
    update!(solver.optimizer, model_params, gs)
end
