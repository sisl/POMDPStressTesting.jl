using Test

testdir = joinpath(dirname(@__DIR__), "test")
cd(testdir)

include("Walk1D.jl")
# (mdp, action_trace, failure_rate) = run_ast()
@test map(a->a.seed, action_trace) == [0x109da377, 0xc1cb6abd, 0x9a6c89a9, 0xedc1929d, 0xa0d4d105, 0x030b59a9, 0xef133c9b, 0xcfabd102]

include("Walk2D.jl")
# (mdp, action_trace, failure_rate) = run_ast()
last_yx = sum(map(a->[a.sample[k].value for k in keys(a.sample)], action_trace))
@test last_yx == [10.667201958002748, 18.47100069435557]