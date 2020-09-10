"""
Adaptive Stress Testing for the POMDPs.jl ecosystem."
"""
module POMDPStressTesting

include("AST.jl")
include(joinpath("solvers", "drl", "DeepRL.jl"))

using .AST
using .DeepRL
using Distributions
using ProgressMeter
using Parameters
using Random
using POMDPs
using MCTS

# Visualization specific.
using PyPlot # TODO. Requires.jl
using Seaborn # for kernel density
using Statistics
using D3Trees

try
    using CrossEntropyMethod # TODO: POMDPs registry?
catch err
    if err isa ArgumentError
        error("Please install CrossEntropyMethod.jl via:\nusing Pkg; pkg\"add https://github.com/mossr/CrossEntropyMethod.jl.git\"")
    end
end


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