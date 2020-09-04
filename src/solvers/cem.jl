# Cross-entropy method, stochastic optimization solver.
@with_kw mutable struct CEMSolver
    n_iterations::Int64 = 100
    episode_length::Int64 = 30
    num_samples::Int64 = 100
    min_elite_samples::Int64 = Int64(floor(0.1*num_samples))
    max_elite_samples::Int64 = typemax(Int64)
    elite_thresh::Float64 = -0.99
    weight_fn::Function = (d,x) -> 1.0
    add_entropy::Function = x->x
    show_progress::Bool = true
    verbose::Bool = false
end


mutable struct CEMPlanner{P<:Union{MDP,POMDP}}
    solver::CEMSolver
    mdp::P
    is_dist::Union{Dict{Symbol,Vector{Sampleable}},Nothing} # optimized importance sample distribution
end


# No work performed.
POMDPs.solve(solver::CEMSolver, mdp::Union{POMDP,MDP}) = CEMPlanner(solver, mdp, nothing)


function cem_losses(d, sample; mdp::ASTMDP, initstate::ASTState)
    sim = mdp.sim
    env = GrayBox.environment(sim)
    s = initstate
    R = 0 # accumulated reward

    BlackBox.initialize!(sim)
    AST.go_to_state(mdp, s)

    sample_length = length(last(first(sample))) # get length of sample vector ("second" element in pair using "first" key)
    for i in 1:sample_length
        env_sample = GrayBox.EnvironmentSample()
        for k in keys(sample)
            value = sample[k][i]
            logprob = logpdf(env[k], value) # log-probability from true distribution
            env_sample[k] = GrayBox.Sample(value, logprob)
        end
        a = ASTSampleAction(env_sample)
        (s, r) = @gen(:sp, :r)(mdp, s, a)
        R += r
        if BlackBox.isterminal(sim)
            break
        end
    end

    return -R # negative (loss)
end


function Base.convert(::Type{Dict{Symbol, Vector{Sampleable}}}, env::GrayBox.Environment, max_steps::Integer)
    dist_vector = Dict{Symbol, Vector{Sampleable}}()
    for k in keys(env)
        dist_vector[k] = fill(env[k], max_steps)
    end
    return dist_vector::Dict{Symbol, Vector{Sampleable}}
end


# Online work performed here.
function POMDPs.action(planner::CEMPlanner, s; rng=Random.GLOBAL_RNG)
    mdp::ASTMDP = planner.mdp
    if actiontype(mdp) != ASTSampleAction
        error("MDP action type must be ASTSampleAction to use CEM.")
    end

    env::GrayBox.Environment = GrayBox.environment(mdp.sim)

    # Importance sampling distributions, fill one per time step.
    is_dist_0 = convert(Dict{Symbol, Vector{Sampleable}}, env, planner.solver.episode_length)

    # Run cross-entropy method using importance sampling
    loss = (d, sample)->cem_losses(d, sample; mdp=mdp, initstate=s)
    is_dist_opt = cross_entropy_method(loss,
                                       is_dist_0;
                                       max_iter=planner.solver.n_iterations,
                                       N=planner.solver.num_samples,
                                       min_elite_samples=planner.solver.min_elite_samples,
                                       max_elite_samples=planner.solver.max_elite_samples,
                                       elite_thresh=planner.solver.elite_thresh,
                                       weight_fn=planner.solver.weight_fn,
                                       add_entropy=planner.solver.add_entropy,
                                       verbose=planner.solver.verbose,
                                       show_progress=planner.solver.show_progress,
                                       rng=rng)

    # Save the importance sampling distributions
    planner.is_dist = is_dist_opt

    # Pass back action trace if recording is on (i.e. top_k)
    if mdp.params.top_k > 0
        return get_top_path(mdp)
    else
        return planner.is_dist # pass back the importance sampling distributions
    end
end


# Search using the planner from initial AST state.
# Pass back best action trace (or importance sampling distribution)
function search!(planner::CEMPlanner)
    mdp::ASTMDP = planner.mdp
    Random.seed!(mdp.params.seed) # Determinism
    s = AST.initialstate(mdp)
    return action(planner, s)
end

