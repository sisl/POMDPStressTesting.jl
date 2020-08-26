using Test

testdir = joinpath(dirname(@__DIR__), "test")
cd(testdir)

@test begin
    @info "Walk1D"
    include("Walk1D.jl")
    true
end
@test begin @info(MCTSASTSolver); (mdp, action_trace, failure_rate) = run_ast(solver=MCTSASTSolver); true end
@test begin @info(CEMSolver); (mdp, action_trace, failure_rate) = run_ast(solver=CEMSolver); true end
@test begin @info(PPOSolver); (mdp, action_trace, failure_rate) = run_ast(solver=PPOSolver); true end
@test begin @info(TRPOSolver); (mdp, action_trace, failure_rate) = run_ast(solver=TRPOSolver); true end
@test begin @info(RandomSearchSolver); (mdp, action_trace, failure_rate) = run_ast(solver=RandomSearchSolver); true end

@test begin
    @info "Walk2D"
    include("Walk2D.jl")
    true
end
@test begin @info(MCTSASTSolver); (mdp, action_trace, failure_rate) = run_ast(solver=MCTSASTSolver); true end
@test begin @info(CEMSolver); (mdp, action_trace, failure_rate) = run_ast(solver=CEMSolver); true end
@test begin @info(PPOSolver); (mdp, action_trace, failure_rate) = run_ast(solver=PPOSolver); true end
@test begin @info(TRPOSolver); (mdp, action_trace, failure_rate) = run_ast(solver=TRPOSolver); true end
@test begin @info(RandomSearchSolver); (mdp, action_trace, failure_rate) = run_ast(solver=RandomSearchSolver); true end
