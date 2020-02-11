using Revise

include("POMDPStressTesting.jl")
using .POMDPStressTesting
# using Distributions

# Simple example: find noise that leads to failure states in stochastic GridWorld (i.e. apply noise to state transitions)

function BlackBox.isevent(s)
    return NaN
end


function BlackBox.miss_distance(s)
	return NaN
end


function BlackBox.transition_model(s)
	return NaN
end


function BlackBox.evaluate(s)
	event::Bool = isevent(s)
	transition_prob::Float64 = transition_model(s)
	return (transition_prob, event)
	# return (transition_prob, miss_distance(s))
end