@with_kw mutable struct RandomSearchSolver
    n_iterations::Int64 = 100
    episode_length::Int64 = 10
    show_progress::Bool = true
end


mutable struct RandomSearchPlanner{P<:Union{MDP,POMDP}}
    solver::RandomSearchSolver
    mdp::P
end


# No work performed.
POMDPs.solve(solver::RandomSearchSolver, mdp::Union{POMDP,MDP}) = RandomSearchPlanner(solver, mdp)


# Online work performed here.
function POMDPs.action(planner::RandomSearchPlanner, s)
    # Pull out variables
    mdp::ASTMDP = planner.mdp
    sim::GrayBox.Simulation = mdp.sim
    n_iterations::Int64 = planner.solver.n_iterations
    depth::Int64 = planner.solver.episode_length
    planner.solver.show_progress ? progress = Progress(n_iterations) : nothing

    # Run simulation
    for i in 1:n_iterations
        BlackBox.initialize!(sim)
        AST.go_to_state(mdp, s)
        AST.rollout(mdp, s, depth) # Evaluate system.
        planner.solver.show_progress ? next!(progress) : nothing
    end

    # Pass back action trace if recording is on (i.e. top_k)
    if mdp.params.top_k > 0
        return get_top_path(mdp)
    end
end


# Search using the planner from the initial AST state.
# Pass back best action trace.
function AST.search!(planner::RandomSearchPlanner)
    mdp::ASTMDP = planner.mdp
    Random.seed!(mdp.params.seed) # Determinism
    s = AST.initialstate(mdp)
    return action(planner, s)
end
