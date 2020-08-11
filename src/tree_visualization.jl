using POMDPPolicies
using D3Trees
using MCTS

 # Full width cells in Jupyter notebook
full_width_notebook() = display(HTML("<style>.container { width:100% !important; }</style>"))

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
function visualize(mdp::AST.ASTMDP, planner::MCTS.DPWPlanner)
    tree = playout(mdp, planner; return_tree=true)
    d3 = visualize(tree)
	return d3::D3Tree
end

function visualize(tree::MCTS.DPWTree)
    d3::D3Tree = D3Tree(tree, init_expand=1)
    return d3::D3Tree
end