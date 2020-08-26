# Modified from Shreyas Kowshik's implementation.

# Returns an episode's worth of experience
# @everywhere # TODO.
function run_episode(env::MDPEnvironment, policy::Union{CategoricalPolicy, DiagonalGaussianPolicy}, num_steps::Int)
    experience = []
    s = reset!(env)
    for i in 1:num_steps
        a = get_action(policy, s)
        ast_action = translate_ast_action(env.problem.sim, a, actiontype(env.problem))
        s′, r, done, _ = step!(env, ast_action)
        push!(experience, (s,a,r,s′))
        s = s′
        if done
            break
        end
    end
    return experience
end


function get_rollouts(env, policy, num_steps::Int)
    g = []
    for w in workers()
        push!(g, run_episode(env, policy, num_steps))
    end
    fetch.(g)
end


"""
Process and extraction information from rollouts
"""
function collect_rollouts!(env, solver, policy, episode_buffer::Buffer, num_steps::Int, stats_buffer::Buffer)
    rollouts = get_rollouts(env, policy, num_steps)

    # Process the variables
    states = []
    actions = []
    rewards = []
    next_states = []
    advantages = []
    returns = []
    log_probs = []
    kl_params = []

    # Logging statistics
    rollout_returns = []

    for ro in rollouts
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []

        for i in 1:length(ro)
            (episode_state, episode_action, episode_reward, episode_next_state) = ro[i]
            push!(episode_states, episode_state)
            push!(episode_actions, episode_action)
            push!(episode_rewards, episode_reward)
            push!(episode_next_states, episode_next_state)

            if typeof(episode_action) <: Int64
                push!(log_probs, log_prob(policy, reshape(episode_state, length(episode_state), 1), [episode_action]).data)
            elseif typeof(episode_action) <: Array
                push!(log_probs, log_prob(policy, reshape(episode_state, length(episode_state), 1), episode_action).data)
            end

            # Kl divergence variables
            if typeof(policy) <: CategoricalPolicy
                push!(kl_params, log_probs[end])
            elseif typeof(policy) <: DiagonalGaussianPolicy
                μ = policy.μ(reshape(episode_state, length(episode_state), 1)).data
                logΣ = policy.logΣ.data
                push!(kl_params, [μ, logΣ])
            end
        end

        episode_advantages = gae(policy, episode_states, episode_actions, episode_rewards, episode_next_states, num_steps; γ=solver.gae_γ, λ=solver.gae_λ)
        episode_advantages = normalize(episode_advantages)

        if solver.terminate_horizon
            episode_returns = disconunted_returns(episode_rewards)
        else
            solver.verbose ? println("\33[2J") : nothing # Clear screen
            solver.verbose ? println("\033[1;1H") : nothing # Move cursor to (1,1)
            solver.verbose ? println("Appending value of last state to returns") : nothing
            episode_returns = disconunted_returns(episode_rewards, policy.value_net(episode_states[end]).data[1])
        end

        push!(states, hcat(episode_states...))
        push!(actions, hcat(episode_actions...))
        push!(rewards, hcat(episode_rewards...))
        push!(next_states, hcat(episode_next_states...))
        push!(advantages, hcat(episode_advantages...))
        push!(returns, hcat(episode_returns...))

        # Variables for logging
        push!(rollout_returns, episode_returns)
    end

    # Normalize advantage across all processes
    # advantages = normalize_across_procs(hcat(advantages...), solver.episode_length)

    episode_buffer.exp_dict["states"] = hcat(states...)
    episode_buffer.exp_dict["actions"] = hcat(actions...)
    episode_buffer.exp_dict["rewards"] = hcat(rewards...)
    episode_buffer.exp_dict["next_states"] = hcat(next_states...)
    episode_buffer.exp_dict["advantages"] = hcat(advantages...)
    episode_buffer.exp_dict["returns"] = hcat(returns...)
    episode_buffer.exp_dict["log_probs"] = hcat(log_probs...)
    episode_buffer.exp_dict["kl_params"] = copy(kl_params)

    # Log the statistics
    add!(stats_buffer, "rollout_rewards", sum(hcat(rewards...)))

    solver.verbose ? println("Rollout rewards : $(sum(hcat(rewards...)))") : nothing
end
