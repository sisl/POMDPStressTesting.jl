"""
Provides virtual interface for the black-box system under test (SUT)
"""
module BlackBox

export
	Simulation,
	initialize,
    isevent,
    miss_distance,
    transition_model,
    evaluate,
    isterminal


"""
	BlackBox.Simulation

Abstract base type for a black-box simulation.
"""
abstract type Simulation end


"""
    initialize(sim::BlackBox.Simulation)

Reset state to its initial state.
"""
function initialize end


"""
    isevent(sim::BlackBox.Simulation)::Bool

Return a boolean indicating if the SUT reached an event of interest.
"""
function isevent end


"""
    miss_distance(sim::BlackBox.Simulation)

Return how close to an event a terminal state was (i.e. some measure of "miss distance" to the event of interest).
"""
function miss_distance end


"""
    transition_model(sim::BlackBox.Simulation)::Tuple(prob, sample)

Return the transition probability and sampled value given the current state [0-1].
"""
function transition_model end


"""
    evaluate(sim::BlackBox.Simulation)::Tuple(transition_probability, isevent, miss_distance)

Evaluate the SUT given some input seed and current state, returns `transition_probability`, `isevent` indication, and `miss_distance`.
"""
function evaluate end


"""
    isterminal(sim::BlackBox.Simulation)::Bool

Return an indication that the simulation is in a terminal state.
"""
function isterminal end


end # module BlackBox