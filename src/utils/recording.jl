"""
    AST.record(::ASTMDP, sym::Symbol, val)

Recard an ASTMetric specified by `sym`.
"""
function record(mdp::ASTMDP, sym::Symbol, val)
    if mdp.params.debug
        push!(getproperty(mdp.metrics, sym), val)
    end
end

function record(mdp::ASTMDP; prob=1, logprob=exp(prob), miss_distance=Inf, reward=-Inf, event=false, terminal=false, rate=-Inf)
    AST.record(mdp, :prob, prob)
    AST.record(mdp, :logprob, logprob)
    AST.record(mdp, :intermediate_logprob, logprob)
    AST.record(mdp, :miss_distance, miss_distance)
    AST.record(mdp, :reward, reward)
    AST.record(mdp, :intermediate_reward, reward)
    AST.record(mdp, :rate, rate)
    AST.record(mdp, :event, event)
    AST.record(mdp, :terminal, terminal)
end

function record_returns(mdp::ASTMDP)
    # compute returns up to now.
    rewards = mdp.metrics.intermediate_reward
    G = returns(rewards, γ=discount(mdp))
    AST.record(mdp, :returns, G)
    mdp.metrics.intermediate_reward = [] # reset
end

function returns(R; γ=1)
    T = length(R)
    G = zeros(T)
    for t in reverse(1:T)
        G[t] = t==T ? R[t] : G[t] = R[t] + γ*G[t+1]
    end
    return G
end

function record_likelihoods(mdp::ASTMDP)
    # compute log-likelihood up to now.
    τ_logprobs = mdp.metrics.intermediate_logprob
    if isempty(τ_logprobs)
        logprobs = -Inf
    else
        logprobs = sum(τ_logprobs)
    end
    AST.record(mdp, :logprobs, logprobs)
    mdp.metrics.intermediate_logprob = [] # reset
end

function combine_ast_metrics(plannervec::Vector)
    return ASTMetrics(
        miss_distance=vcat(map(planner->planner.mdp.metrics.miss_distance, plannervec)...),
        rate=vcat(map(planner->planner.mdp.metrics.rate, plannervec)...),
        logprob=vcat(map(planner->planner.mdp.metrics.logprob, plannervec)...),
        intermediate_logprob=vcat(map(planner->planner.mdp.metrics.intermediate_logprob, plannervec)...),
        logprobs=vcat(map(planner->planner.mdp.metrics.logprobs, plannervec)...),
        prob=vcat(map(planner->planner.mdp.metrics.prob, plannervec)...),
        reward=vcat(map(planner->planner.mdp.metrics.reward, plannervec)...),
        intermediate_reward=vcat(map(planner->planner.mdp.metrics.intermediate_reward, plannervec)...),
        returns=vcat(map(planner->planner.mdp.metrics.returns, plannervec)...),
        event=vcat(map(planner->planner.mdp.metrics.event, plannervec)...),
        terminal=vcat(map(planner->planner.mdp.metrics.terminal, plannervec)...))
end