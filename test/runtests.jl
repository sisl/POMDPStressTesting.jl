using Test
using NBInclude

testdir = joinpath(dirname(@__DIR__), "test")
notebookdir = joinpath(dirname(@__DIR__), "notebooks")

function test_solvers(solvers=[MCTSPWSolver, CEMSolver, PPOSolver, TRPOSolver, RandomSearchSolver]; skip_trpo=false)
    for solver in solvers
        @test begin
            if skip_trpo && solver == TRPOSolver
                @info "Skipping TRPOSolver."
            else
                @info solver
                run_ast(solver=solver)
            end
            true
        end
    end
end


@test begin
    @info "Walk1D Jupyter notebook"
    @nbinclude(joinpath(notebookdir, "Walk1D.ipynb"))
    true
end


cd(testdir)


@test begin
    @info "Walk1D"
    include("Walk1D.jl")
    true
end


@test begin
    @info "Extra functions test"
    (planner, _, _) = run_ast(solver=MCTSPWSolver)
    d3tree = visualize(planner)
    action_trace = search!(planner; verbose=true)
    actions = online_path(planner.mdp, planner)
    x_trace = playback(planner, actions, sim->sim.x; return_trace=true)
    final_state = playback(planner, actions, sim->sim.x)
    AST.state_trace(final_state)
    AST.q_trace(final_state)
    reset_metrics!(planner.mdp)

    # Policy saving
    solver = PPOSolver(num_episodes=100, episode_length=30, save=true, verbose=true)
    planner = solve(solver, planner.mdp)
    search!(planner)

    # Policy loading
    solver.resume = true
    search!(planner)

    # No top_k action trace
    solver.resume = false
    planner.mdp.params.top_k = 0
    ast_action = search!(planner)
    true
end


# Walk1D with different solvers
test_solvers()


@test begin
    @info "Walk2D"
    include("Walk2D.jl")
    true
end
test_solvers(skip_trpo=true) # TRPO slows down the entire test suite, so skip it for Walk2D (coverd by Walk1D)


@test begin
    @info "EpisodicWalk1D"
    include("EpisodicWalk1D.jl")
    true
end


@test begin
    @info "CategoricalWalk1D"
    include("CategoricalWalk1D.jl")

    # Policy saving
    (planner, _, _) = run_ast(solver=PPOSolver)
    solver = PPOSolver(num_episodes=100, episode_length=30, save=true, verbose=true, policy_type=:discrete)
    planner = solve(solver, planner.mdp)
    search!(planner)

    # Policy loading
    solver.resume = true
    search!(planner)

    true
end
