"""
Provides virtual interface for the black-box system under test (SUT)
"""
module BlackBox

export
    isevent,
    miss_distance,
    transition_prob,
    evaluate


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
    transition_prob()::Float64

Return the transition probability of the current state [0-1].
"""
function transition_prob end

"""
    evaluate(a::Seed)

Evaluate the SUT given some input seed.
"""
function evaluate end


end # module BlackBox