# using Revise

# include("POMDPStressTesting.jl")
# using .POMDPStressTesting

include("Walk1D.jl")

mdp = runtest()

using POMDPs
using MCTS

@requirements_info MCTSSolver() mdp