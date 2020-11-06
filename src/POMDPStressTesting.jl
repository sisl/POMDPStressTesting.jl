"""
Adaptive Stress Testing for the POMDPs.jl ecosystem.
"""
module POMDPStressTesting

include("AST.jl")
include(joinpath("solvers", "drl", "PolicyOptimization.jl"))

using .AST
using .PolicyOptimization
using CrossEntropyMethod
using Distributions
using MCTS
using Parameters
using POMDPs
using ProgressMeter
using Random

# Visualization specific.
using D3Trees
using PyPlot # TODO. Requires.jl
using Seaborn # for kernel density
using Statistics


export AST,
       ASTMDP,
       ASTParams,
       ASTState,
       ASTAction,
       ASTSeedAction,
       ASTSampleAction,
       actiontype,
       BlackBox,
       GrayBox,

       search!,
       playback,
       online_path,
       get_top_path,
       hash_uint32,

       visualize,
       full_width_notebook,
       episodic_figures,
       distribution_figures,
       print_metrics,
       reset_metrics!,

       MCTSPWSolver,
       CEMSolver,
       CEMPlanner,
       RandomSearchSolver,
       RandomSearchPlanner,
       TRPOSolver,
       TRPOPlanner,
       PPOSolver,
       PPOPlanner,

       # For MCTS and RandomSearch
       solve,
       action


include(joinpath("utils", "metrics.jl"))
include(joinpath("visualization", "tree_visualization.jl"))
include(joinpath("visualization", "figures.jl"))

include(joinpath("solvers", "mcts.jl"))
include(joinpath("solvers", "cem.jl"))
include(joinpath("solvers", "random_search.jl"))

end # module POMDPStressTesting