"""
Provides virtual interface for the black-box system under test (SUT)
"""
module BlackBox

export
    Simulation,
    initialize!,
    transition!,
    evaluate!,
    distance!,
    isevent!,
    isterminal!


"""
    BlackBox.Simulation

Abstract base type for a black-box simulation.
"""
abstract type Simulation end


"""
    initialize!(sim::BlackBox.Simulation)

Reset state to its initial state.
"""
function initialize!(sim::Simulation)::Nothing end


"""
    evaluate!(sim::BlackBox.Simulation)::Tuple(logprob, isevent, miss_distance)

Evaluate the SUT given some input seed and current state, returns `logprob`, `isevent` indication, and `miss_distance`.
"""
function evaluate!(sim::Simulation)::Tuple{Real, Real, Bool} end


"""
    transition!(sim::BlackBox.Simulation)::Real

Return the transition log-probability and sampled value given the current state.
"""
function transition!(sim::Simulation)::Real end


"""
    distance!(sim::BlackBox.Simulation)::Real

Return how close to an event a terminal state was (i.e. some measure of "miss distance" to the event of interest).
"""
function distance!(sim::Simulation)::Real end


"""
    isevent!(sim::BlackBox.Simulation)::Bool

Return a boolean indicating if the SUT reached an event of interest.
"""
function isevent!(sim::Simulation)::Bool end


"""
    isterminal!(sim::BlackBox.Simulation)::Bool

Return an indication that the simulation is in a terminal state.
"""
function isterminal!(sim::Simulation)::Bool end


end # module BlackBox