# Modified from Shreyas Kowshik's implementation.

"""
Categorical policy, for discrete action spaces.
"""
mutable struct CategoricalPolicy
    π # Neural network for the policy
    value_net # Neural network for the value function
    solver # Reference to the TRPOSolver or PPOSolver
end

function CategoricalPolicy(solver)
    policy_net = Chain(Dense(solver.state_size, solver.hidden_layer_size, relu; initW=_random_normal, initb=constant_init),
                       Dense(solver.hidden_layer_size, solver.ACTION_SIZE; initW=_random_normal, initb=constant_init),
                       x -> softmax(x))

    value_net = Chain(Dense(solver.state_size, solver.hidden_layer_size ,relu; initW=_random_normal),
                      Dense(solver.hidden_layer_size, solver.hidden_layer_size, relu; initW=_random_normal),
                      Dense(solver.hidden_layer_size, 1; initW=_random_normal))

    return CategoricalPolicy(policy_net, value_net, solver)
end


"""
Diagonal Gaussian policy, for continuous action space.
"""
mutable struct DiagonalGaussianPolicy
    μ # Neural network for the mean of the Gaussian distribution
    logΣ # Neural network for the log standard deviation of the Gaussian distribution
    value_net # Neural network for the value function
    solver # Reference to the TRPOSolver or PPOSolver
end

function DiagonalGaussianPolicy(solver, log_std)
    μ = Chain(Dense(solver.state_size, solver.hidden_layer_size, tanh; initW=_random_normal, initb=constant_init),
              Dense(solver.hidden_layer_size, solver.action_size; initW=_random_normal, initb=constant_init),
              x->tanh.(x),
              x->param(2.0) .* x) # TODO. Make this vector of environment STDs (i.e. scale it to [-x, +x] action bound)

    value_net = Chain(Dense(solver.state_size, solver.hidden_layer_size, tanh; initW=_random_normal),
                      Dense(solver.hidden_layer_size, solver.hidden_layer_size, tanh; initW=_random_normal),
                      Dense(solver.hidden_layer_size, 1; initW=_random_normal))

    logΣ = param(ones(solver.action_size) * log_std)

    return DiagonalGaussianPolicy(μ, logΣ, value_net, solver)
end


"""
Takes in the policy and returns an action based on the current state.
"""
function get_action(policy::CategoricalPolicy, state)
    action_probs = policy.π(state).data
    action_probs = reshape(action_probs, policy.solver.action_size)
    return Distributions.sample(1:policy.solver.action_size, Distributions.Weights(action_probs))
end


function get_action(policy::DiagonalGaussianPolicy, state)
    # Our policy outputs the parameters of a Normal distribution
    μ = reshape(policy.μ(state), policy.solver.action_size)
    log_std = policy.logΣ

    σ² = (exp.(log_std)).^2
    Σ = diagm(0=>σ².data)
    distribution = MvNormal(μ.data, Σ)

    return rand(distribution, policy.solver.action_size)
end


test_action(policy::CategoricalPolicy, state) = get_action(policy, state)
test_action(policy::DiagonalGaussianPolicy, state) = policy.μ(state).data # Use only the mean for prediction


"""
Translate action from neural network policy to either an ASTSeedAction or an ASTSampleAction
"""
function translate_ast_action(sim::GrayBox.Simulation, action, ::Type{ASTSeedAction})
    # Convert to seed: NOTE, not every useful when using PPO---effectively random. You've been warned (try MCTS)!
    return ASTSeedAction(hash_uint32(action...))
end


function translate_ast_action(sim::GrayBox.Simulation, action, ::Type{ASTSampleAction})
    gray_environment = GrayBox.environment(sim)
    environment_sample = GrayBox.EnvironmentSample()
    for (i,k) in enumerate(keys(gray_environment))
        # log-probability from the environment's distributions (not the log_prob from the NN policy)
        logp = logpdf(gray_environment[k], action[i])
        environment_sample[k] = GrayBox.Sample(action[i], logp)
    end
    return ASTSampleAction(environment_sample)
end


"""
Return the log-probability of an action under the current policy parameters.
"""
function log_prob(policy::CategoricalPolicy, states::Array, actions::Array)
    action_probs = policy.π(states)
    actions_one_hot = zeros(policy.solver.action_size, size(action_probs)[end])
    for i in 1:size(action_probs)[end]
        actions_one_hot[actions[:,i][1],i] = 1.0
    end
    return log.(sum((action_probs .+ 1f-5) .* actions_one_hot, dims=1))
end


function log_prob(policy::DiagonalGaussianPolicy, states::Array, actions::Array)
    μ = policy.μ(states)
    σ = exp.(policy.logΣ)
    σ² = σ.^2
    log_probs = broadcast(-, ((actions .- μ).^2)./(2 .* σ²)) .- 0.5*log.(sqrt(2π)) .- log.(σ)
    return log_probs
end


"""
Return the entropy of the policy distribution.
"""
function entropy(policy::CategoricalPolicy, states::Array)
    action_probs = policy.π(states)
    return sum(action_probs .* log.(action_probs .+ 1f-10), dims=1)
end

entropy(policy::DiagonalGaussianPolicy, states::Array) = 0.5 + 0.5 * log(2π) .+ policy.logΣ


"""
kl_params:

`old_log_probs` : CategoricalPolicy

`Array([μ, logΣ])` : DiagonalGaussianPolicy
"""
function kl_divergence(policy::CategoricalPolicy, kl_params, states::Array)
    old_log_probs = hcat(cat(kl_params..., dims=1)...)

    action_probs = policy.π(states)
    log_probs = log.(action_probs)

    log_ratio = log_probs .- old_log_probs
    kl_div = (exp.(old_log_probs)) .* log_ratio

    return -1.0f0 .* sum(kl_div, dims=1)
end


function kl_divergence(policy::DiagonalGaussianPolicy, kl_params, states::Array)
    μ0 = policy.μ(states)
    logΣ0 = policy.logΣ
    μ1 = mus = hcat([kl_params[i][1] for i in 1:length(kl_params)]...)
    logΣ1 = hcat([kl_params[i][2] for i in 1:length(kl_params)]...)

    var0 = exp.(2 .* logΣ0)
    var1 = exp.(2 .* logΣ1)
    pre_sum = 0.5 .* (((μ0 .- μ1).^2 .+ var0) ./ (var1 .+ 1e-8) .- 1.0f0) .+ logΣ1 .- logΣ0
    kl = sum(pre_sum, dims=1)

    return kl
end


get_policy_params(policy::CategoricalPolicy) = Flux.params(policy.π)
get_policy_params(policy::DiagonalGaussianPolicy) = Flux.params(Flux.params(policy.μ)..., Flux.params(policy.logΣ)...)

get_policy_net(policy::CategoricalPolicy) = [policy.π]
get_policy_net(policy::DiagonalGaussianPolicy) = [policy.μ, policy.logΣ]

get_value_params(policy::Union{CategoricalPolicy, DiagonalGaussianPolicy}) = Flux.params(policy.value_net)

get_value_net(policy::Union{CategoricalPolicy, DiagonalGaussianPolicy}) = [policy.value_net]


function get_policy(solver)
    if solver.policy_type == :discrete
        return CategoricalPolicy(solver)
    elseif solver.policy_type == :continuous
        return DiagonalGaussianPolicy(solver, solver.log_std)
    else
        error("Policy type not supported, $(solver.policy_type)")
    end
end


# TODO. Generalize for discrete spaces.
function set_action_size!(solver, mdp::ASTMDP)
    if actiontype(mdp)== ASTSeedAction
        solver.action_size = 1
    elseif actiontype(mdp) == ASTSampleAction
        solver.action_size = length(GrayBox.environment(mdp.sim))
    end
    return solver
end