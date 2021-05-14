import .AST: ASTMDP, ASTMetrics

"""
Collection of failure related metrics.
- iteration of first failure
- number of failures
- failure rate
- highest log-likelihood of failure
"""
@with_kw struct FailureMetrics
    num_terminals = NaN
    first_failure = NaN
    num_failures = NaN
    failure_rate = NaN
    highest_loglikelihood = NaN
end


"""
Calculate highest log-likelihood of failure.
"""
highest_loglikelihood_of_failure(planner) = highest_loglikelihood_of_failure(planner.mdp.metrics)
function highest_loglikelihood_of_failure(metrics::ASTMetrics)
    # TODO: get this index from the returned `action_trace` itself
    failure_loglikelihood = NaN
    if any(metrics.event)
        # Failures were found.
        failure_loglikelihood = maximum(metrics.logprob[metrics.event])
    end
    return failure_loglikelihood
end


"""
    failure_metrics(planner)
    failure_metrics(mdp::ASTMDP)
    failure_metrics(metrics::ASTMetrics)

Collect failure metrics including:
- iteration of first failure
- number of failures
- failure rate
- highest log-likelihood of failure
"""
failure_metrics(planner) = failure_metrics(planner.mdp.metrics)
failure_metrics(mdp::ASTMDP) = failure_metrics(mdp.metrics)
function failure_metrics(metrics::ASTMetrics)
    E = metrics.event

    if findfirst(E) === nothing
        @info "No failures recorded."
        return 0
    else
        terminal_idx = findall(metrics.terminal)
        if isempty(terminal_idx)
            @warn "No terminal states. Something may be incorrect in simulation."
        else
            possible_failures = metrics.event[terminal_idx] # 0 or 1 if the terminal state was a failure.
            first_failure = findfirst(possible_failures)
            num_terminals = length(terminal_idx)
            num_failures = sum(possible_failures)
            failure_rate = num_failures/num_terminals * 100
            highest_loglikelihood = highest_loglikelihood_of_failure(metrics)
            failure_metrics = FailureMetrics(num_terminals, first_failure, num_failures, failure_rate, highest_loglikelihood)
            return failure_metrics
        end
    end
end


"""
    print_metrics(planner)
    print_metrics(mdp::ASTMDP)
    print_metrics(metrics::ASTMetrics)

Print failure metrics including:
- iteration of first failure
- number of failures
- failure rate
- highest log-likelihood of failure
"""
print_metrics(planner) = print_metrics(planner.mdp.metrics)
print_metrics(mdp::ASTMDP) = print_metrics(mdp.metrics)
function print_metrics(metrics::ASTMetrics)
    fail_metrics = failure_metrics(metrics)
    if fail_metrics isa FailureMetrics
        println("First failure: ", fail_metrics.first_failure, " of ", fail_metrics.num_terminals)
        println("Number of failures: ", fail_metrics.num_failures)
        println("Failure rate: ", round(fail_metrics.failure_rate, digits=5), "%")
        println("Highest log-likelihood of failure: ", round(fail_metrics.highest_loglikelihood_of_failure, digits=5))
    end
    return fail_metrics
end


"""
Display failure metrics in a LaTeX enviroment.
Useful in Pluto.jl notebooks.
"""
function latex_metrics(metrics::FailureMetrics)
    # Note indenting is important here to render correctly.
    return Markdown.parse(string("
\$\$\\begin{align}",
"p(\\text{fail}) &=", metrics.failure_rate/100, "\\\\",
"\\text{first failure index} &= ", metrics.first_failure, "\\text{ of }", metrics.num_terminals, "\\\\",
"\\text{highest log-likelihood of failure} &= ", round(metrics.highest_loglikelihood, digits=5), "\\\\",
"\\end{align}\$\$"))
end
