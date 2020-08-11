"""
    AST.record(::ASTMDP, sym::Symbol, val)

Recard an ASTMetric specified by `sym`.
"""
function record(mdp::ASTMDP, sym::Symbol, val)
    if mdp.params.debug
        push!(getproperty(mdp.metrics, sym), val)
    end
end

function record(mdp::ASTMDP; prob=1, logprob=exp(prob), miss_distance=Inf, reward=-Inf)
    AST.record(mdp, :prob, prob)
    AST.record(mdp, :logprob, logprob)
    AST.record(mdp, :miss_distance, miss_distance)
    AST.record(mdp, :reward, reward)
end
