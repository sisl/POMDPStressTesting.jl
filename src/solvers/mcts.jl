# Wrapper: AST version of this the MCTS solver (i.e. sets required parameters)
function MCTSASTSolver(; kwargs...)
    return MCTS.DPWSolver(; estimate_value=AST.rollout, # required.
                            enable_state_pw=false, # required.
                            reset_callback=AST.go_to_state, # Custom fork of MCTS.jl
                            tree_in_info=true,
                            kwargs...)
end
