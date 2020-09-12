"""
Provides virtual interface for the black-box system under test (SUT)
"""
module BlackBox

include("GrayBox.jl")

export initialize!,
       evaluate!,
       distance,
       isevent,
       isterminal


"""
    initialize!(sim::GrayBox.Simulation)

Reset state to its initial state.
"""
function initialize!(sim::GrayBox.Simulation)::Nothing end


"""
    evaluate!(sim::GrayBox.Simulation)::Tuple{logprob::Real, miss_distance::Real, isevent::Bool}
    evaluate!(sim::GrayBox.Simulation, sample::GrayBox.EnvironmentSample)::Tuple{logprob::Real, miss_distance::Real, isevent::Bool}

Evaluate the SUT given some input seed and current state, returns `logprob`, `miss_distance`, and `isevent` indication.
If the `sample` version is implemented, then ASTSampleAction will be used instead of ASTSeedAction.
"""
function evaluate!(sim::GrayBox.Simulation)::Tuple{Real, Real, Bool} end
function evaluate!(sim::GrayBox.Simulation, sample::GrayBox.EnvironmentSample)::Tuple{Real, Real, Bool} end



"""
    distance(sim::GrayBox.Simulation)::Real

Return how close to an event a terminal state was (i.e. some measure of "miss distance" to the event of interest).
"""
function distance(sim::GrayBox.Simulation)::Real end


"""
    isevent(sim::GrayBox.Simulation)::Bool

Return a boolean indicating if the SUT reached an event of interest.
"""
function isevent(sim::GrayBox.Simulation)::Bool end


"""
    isterminal(sim::GrayBox.Simulation)::Bool

Return an indication that the simulation is in a terminal state.
"""
function isterminal(sim::GrayBox.Simulation)::Bool end


end # module BlackBox