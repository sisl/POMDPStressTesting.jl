using Revise

include("POMDPStressTesting.jl")
using .POMDPStressTesting

function BlackBox.isevent(x)
    @show x
    return x > 0
end

@show BlackBox.isevent(1)