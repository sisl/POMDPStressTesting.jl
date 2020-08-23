# Wrapper: AST version of this the MCTS solver (i.e. sets required parameters)
function MCTSASTSolver(; kwargs...)
    try
        return MCTS.DPWSolver(; estimate_value=AST.rollout, # required.
                                enable_state_pw=false, # required.
                                reset_callback=AST.go_to_state, # Custom fork of MCTS.jl
                                tree_in_info=true,
                                kwargs...)
    catch err
        if err isa MethodError
            error("Please install MCTS.jl fork via:\nusing Pkg; Pkg.add(PackageSpec(url=\"https://github.com/mossr/MCTS.jl.git\"))")
        else
            throw(err)
        end
    end
end
