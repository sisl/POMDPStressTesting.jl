# Modified from Shreyas Kowshik's implementation.

function train_step!(planner::Union{TRPOPlanner, PPOPlanner},
                     policy::Union{CategoricalPolicy, DiagonalGaussianPolicy},
                     episode_buffer::Buffer, stats_buffer::Buffer)
    env = planner.env
    solver = planner.solver

    clear!(episode_buffer)
    collect_rollouts!(env, solver, policy, episode_buffer, solver.episode_length, stats_buffer)

    idxs = partition(shuffle(1:size(episode_buffer.exp_dict["states"])[end]), solver.batch_size)

    if solver isa TRPOSolver
        θ, reconstruct = get_flat_params(get_policy_net(policy))
        old_params = copy(θ)
    end

    for i in idxs
        mb_states = episode_buffer.exp_dict["states"][:,i]
        mb_actions = episode_buffer.exp_dict["actions"][:,i]
        mb_advantages = episode_buffer.exp_dict["advantages"][:,i]
        mb_returns = episode_buffer.exp_dict["returns"][:,i]
        mb_log_probs = episode_buffer.exp_dict["log_probs"][:,i]
        mb_kl_vars = episode_buffer.exp_dict["kl_params"][i]

        if solver isa TRPOSolver
            trpo_update!(solver, policy, mb_states, mb_actions, mb_advantages, mb_returns, mb_log_probs, mb_kl_vars, old_params, reconstruct)
        elseif solver isa PPOSolver
            kl_div = mean(kl_divergence(policy, mb_kl_vars, mb_states))
            solver.verbose ? println("KL Sample : $(kl_div)") : nothing

            ppo_update!(solver, policy, mb_states, mb_actions, mb_advantages, mb_returns, mb_log_probs, mb_kl_vars)
        end
    end
end


function train!(planner::Union{TRPOPlanner, PPOPlanner})
    solver::Union{TRPOSolver, PPOSolver} = planner.solver

    # Create or load policy
    if solver.resume
        if solver.policy_type == :discrete
            policy = load_policy(solver, "weights", CategoricalPolicy) # TODO: parameterize path
        elseif solver.policy_type == :continuous
            policy = load_policy(solver, "weights", DiagonalGaussianPolicy) # TODO: parameterize path
        end
    else
        policy = get_policy(solver)
    end

    # Define buffers
    episode_buffer::Buffer = initialize_episode_buffer()
    stats_buffer::Buffer = initialize_stats_buffer()
    solver.show_progress ? progress = Progress(solver.num_episodes) : nothing

    for i in 1:solver.num_episodes
        solver.verbose ? println("Episode $i") : nothing
        train_step!(planner, policy, episode_buffer, stats_buffer)
        solver.verbose ? println(mean(stats_buffer.exp_dict["rollout_rewards"])) : nothing
        solver.show_progress ? next!(progress) : nothing

        if solver.save && i % solver.save_frequency == 0
            save_policy(policy, "weights") # TODO. Handle where to save policy
        end
    end

    planner.policy = policy

    return policy
end
