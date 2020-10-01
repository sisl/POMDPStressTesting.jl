# Modified from Shreyas Kowshik's implementation.

# Weight initialization
_random_normal(shape...) = map(Float32, rand(Normal(0, 0.1), shape...))
constant_init(shape...) = map(Float32, ones(shape...) * 0.1)
normalize(arr) = (arr .- mean(arr))./(sqrt(var(arr) + 1e-10))

function normalize_across_procs(arr, episode_length)
    arr = reshape(arr, episode_length, NUM_PROCESSES)
    reshape((arr .- mean(arr,dims=2))./(sqrt.(var(arr, dims=2) .+ 1e-10)), 1, episode_length*NUM_PROCESSES)
end



### Utility functions specific to PPO


"""
Returns a Generalized Advantage Estimate for an episode
"""
function gae(policy, states::Array, actions::Array, rewards::Array, next_states::Array, num_steps::Int; γ=0.99, λ=0.95)
    Â = []
    A = 0.0
    for i in reverse(1:length(states))
        if length(states) < num_steps && i == length(states)
            δ = rewards[i] - policy.value_net(states[i])[1]
        else
            δ = rewards[i] + γ*policy.value_net(next_states[i])[1] - policy.value_net(states[i])[1]
        end

        A = δ + (γ*λ*A)
        push!(Â, A)
    end

    return reverse(Â)
end


"""
Returns the cumulative discounted returns for each timestep
"""
function disconunted_returns(rewards::Array, last_val=0; γ=0.99)
    r = 0.0
    returns = []

    for i in reverse(1:length(rewards))
        r = rewards[i] + γ*r  # TODO. γ^i (?)
        if i == length(rewards)
            r = r + last_val
        end
        push!(returns, r)
    end

    return reverse(returns)
end



### Utility functions specific to TRPO



"""
Obtain the gradient vector product. Intermediate utility function, calculates `Σ∇D_kl*x`

`x` : Variable to be estimated using conjugate gradient `(Hx = g); (NUM_PARAMS,1)`
"""
function gvp(policy, states, kl_vars, x)
    model_params = get_policy_params(policy)
    gs = gradient(() -> kl_loss(policy, states, kl_vars), model_params)

    flat_grads = get_flat_grads(gs, get_policy_net(policy))
    return sum(x' * flat_grads)
end


"""
Computes the Hessian vector product.
Hessian is that of the kl divergence between the old and the new policies w.r.t. the policy parameters.

Returns : `Hx; H = ∇²D_kl`
"""
function Hvp(policy, states, kl_vars, x; damping_coeff=0.1)
    hessian = Zygote.forward_jacobian((x) -> gvp(policy, states, kl_vars, x), x)[2]
    return get_flat_grads(hessian, get_policy_net(policy)) # .+ (damping_coeff .* x)
end


"""
Compute the conjugate gradient.

`b` : Array of shape `(NUM_PARAMS,1)`

Solves `x` for `Hx = b`
"""
function conjugate_gradients(policy, states, kl_vars, b, nsteps=10, err=1e-10)
    x = zeros(size(b))
    r = copy(b)
    p = copy(b)
    rdotr = r' * r

    for i in 1:nsteps
        hvp = Hvp(policy, states, kl_vars, p) # Returns array of shape (NUM_PARAMS,1)

        α = rdotr ./ (p' * hvp)

        x = x .+ (α .* p)
        r = r .- (α .* hvp)

        new_rdotr = r' * r
        β = new_rdotr ./ rdotr
        p = r .+ (β .* p)

        rdotr = new_rdotr

        if rdotr[1] < err
            break
        end
    end

    return x
end



"""
Flattens out the gradients and concatenates them.

`models` : An array of models whose parameter `gradients` are to be falttened.

Returns : Array of shape `(NUM_PARAMS,1)`
"""
function get_flat_grads(gradients, models)
    return collect(Iterators.flatten(map(m->Flux.destructure(m)[1], models)))
end


"""
Flattens out the parameters and concatenates them.

`models` : An array of models whose parameters are to be flattened

Returns : Array of shape `(NUM_PARAMS,1)`
"""
function get_flat_params(models)
    return Flux.destructure(models)
end


"""
Sets values of `parameters` to the `model`.

`parameters` : flattened out array of model parameters.

`models` : an array of models whose parameters are to be set.

`reconstruct` : function output by `Flux.destructure` to reconstruct model from parameters
"""
function set_flat_params!(parameters, models, reconstruct)
    models[:] = reconstruct(parameters)
end

# To avoid mutation
Zygote.@nograd get_flat_params
Zygote.@nograd set_flat_params!
Zygote.@nograd get_flat_grads
