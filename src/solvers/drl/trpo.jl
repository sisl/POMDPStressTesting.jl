# Modified from Shreyas Kowshik's implementation.

# NOTE: TRPO will not work with categorical policies as nested AD is not currently defined for `softmax`
@with_kw mutable struct TRPOSolver
    episode_length::Int64 = 100 # Length of each episode
    terminate_horizon::Bool = false # Zero reward at end of horizon
    resume::Bool = false # Load saved policy and resume.

    # Optimization parameters
    η::Float64 = 1e-3 # learning rate η
    optimizer = ADAM(η) # Optimizer for the value function neural network
    δ = 0.01 # KL-divergence constraint
    value_iterations = 5 # Number of iterations to train the value function network

    # Generalized advantage estimate (GAE) parameters
    gae_γ::Float64 = 0.99 # GAE-Gamma
    gae_λ::Float64 = 0.95 # GAE-Lambda

    # Training parameters
    num_episodes::Int64 = 10 # Number of epochs of interaction (equal to number of policy updates)
    batch_size::Int64 = episode_length

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


mutable struct TRPOPlanner{P<:Union{MDP,POMDP}}
    solver::TRPOSolver
    mdp::P
    env::Union{MDPEnvironment,POMDPEnvironment}
    policy::Union{CategoricalPolicy,DiagonalGaussianPolicy,Nothing}
end


# Uses RLInterface environment wrapper
POMDPs.solve(solver::TRPOSolver, mdp::MDP) = TRPOPlanner(set_action_size!(solver, mdp), mdp, MDPEnvironment(mdp), nothing)
POMDPs.solve(solver::TRPOSolver, mdp::POMDP) = TRPOPlanner(set_action_size!(solver, mdp), mdp, POMDPEnvironment(mdp), nothing)


# NOTE: action and playback shared between TRPO and PPO (see PolicyOptimization.jl)


function policy_loss(policy, states::Array, actions::Array, advantages::Array, old_log_probs::Array)
    # Surrogate loss computation
    new_log_probs = log_prob(policy, states, actions)
    ratio = exp.(new_log_probs .- old_log_probs)
    π_loss = mean(ratio .* advantages)
    return π_loss
end

kl_loss(policy, states::Array, kl_vars) = mean(kl_divergence(policy, kl_vars, states))

value_loss(policy, states::Array, returns::Array) = mean((policy.value_net(states) .- returns).^2)


function linesearch!(solver, policy, step_dir, states, actions, advantages, old_log_probs, kl_vars, old_params, num_steps=10; α=0.5)
    old_loss = policy_loss(policy, states, actions, advantages, old_log_probs).data

    for i in 1:num_steps
        # Obtain new parameters
        new_params = old_params .+ (α^i .* step_dir)

        # Set the new parameters to the policy
        set_flat_params!(new_params, get_policy_net(policy))

        # Compute surrogate loss
        new_loss = policy_loss(policy, states, actions, advantages, old_log_probs).data

        # Compute kl divergence
        kl_div = kl_loss(policy, states, kl_vars).data

        # Output Statistics
        solver.verbose ? println("Old Loss : $old_loss") : nothing
        solver.verbose ? println("New Loss : $new_loss") : nothing
        solver.verbose ? println("KL Div : $kl_div") : nothing

        if new_loss >= old_loss && (kl_div <= solver.δ)
            solver.verbose ? println("Success.") : nothing
            set_flat_params!(new_params, get_policy_net(policy))
            return nothing
        end
    end

    set_flat_params!(old_params, get_policy_net(policy))
end


function trpo_update!(solver, policy, states, actions, advantages, returns, log_probs, kl_vars, old_params)
    model_params = get_policy_params(policy)
    policy_grads = Tracker.gradient(() -> policy_loss(policy, states, actions, advantages, log_probs), model_params)
    flat_policy_grads = get_flat_grads(policy_grads, get_policy_net(policy)).data

    x = conjugate_gradients(policy, states, kl_vars, flat_policy_grads, 10)
    solver.verbose ? println(minimum(x' * Hvp(policy, states, kl_vars, x))) : nothing

    step_dir = nothing
    try
        step_dir = sqrt.((2solver.δ) ./ (x' * Hvp(policy, states, kl_vars, x))) .* x
    catch err
        if err isa DomainError
            @info "Square root of a negative number received...Skipping update"
            return
        else
            throw(err)
        end
    end

    # Do a line search and update the parameters
    linesearch!(solver, policy, step_dir, states, actions, advantages, log_probs, kl_vars, old_params)

    # Update value function
    for _ in 1:solver.value_iterations
        value_params = get_value_params(policy)
        gs = Tracker.gradient(() -> value_loss(policy, states, returns), value_params)
        update!(solver.optimizer, value_params, gs)
    end
end
