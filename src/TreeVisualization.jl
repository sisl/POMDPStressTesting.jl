module TreeVisualization

export visualize

using POMDPStressTesting
using POMDPPolicies
using MCTS
using D3Trees

# Display of action nodes.
function MCTS.node_tag(s::AST.ASTState)
	state_str::String = "0x"*string(s.hash, base=16)
    if s.terminal
        return "Terminal [$state_str]."
    else
        return state_str
    end
end

# Display of state nodes.
seeds2string(a) = join(map(s->"0x" * string(s, base=16), a.rsg.state), ",\n")
function MCTS.node_tag(a::AST.ASTAction)
    if a == action # selected optimal action
        return "—[$(seeds2string(a))]—"
    else
        return "[$(seeds2string(a))]"
    end
end


"""
Visualize MCTS tree structure for AST MDP.
"""
function visualize(mdp::AST.ASTMDP, planner::DPWPlanner)
	state::AST.ASTState = AST.initialstate(mdp)
	(action::AST.ASTAction, info) = action_info(planner, state, tree_in_info=true)
	d3::D3Tree = D3Tree(info[:tree], init_expand=1)

	action_path::Vector{AST.ASTAction} = AST.get_optimal_path(mdp, info[:tree], AST.initialstate(mdp), verbose=true)

	return d3::D3Tree
end


end # module TreeVisualization