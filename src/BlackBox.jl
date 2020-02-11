"""
Provides virtual interface for the black-box system under test (SUT)
"""
module BlackBox

export
	initialize,
    isevent,
    miss_distance,
    transition_model,
    evaluate,
    isterminal


"""
    initialize()

Reset state to its initial state.
"""
function initialize end


"""
    isevent()::Bool

Return a boolean indicating if the SUT reached an event of interest.
"""
function isevent end


"""
    miss_distance()::T

Return how close to an event a terminal state was (i.e. some measure of "miss distance" to the event of interest).
"""
function miss_distance end


"""
    transition_model()::Float64

Return the transition probability of the current state [0-1].
"""
function transition_model end


"""
    evaluate(s::State, a::Seed)::Tuple(transition_probability, isevent)

Evaluate the SUT given some input seed and current state, returns `transition_probability` and `isevent` indication.
"""
function evaluate end


"""
    isterminal()::Bool

Return an indication that the simulation is in a terminal state.
"""
function isterminal end


end # module BlackBox