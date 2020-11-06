"""
Provides virtual interface for the gray-box environment and simulator.
"""
module GrayBox

using Random
using Distributions

export Simulation,
       Sample,
       Environment,
       EnvironmentSample,
       environment,
       transition!


"""
    GrayBox.Simulation

Abstract base type for a gray-box simulation.
"""
abstract type Simulation end



"""
    GrayBox.Sample

Holds sampled value and log-probability.
"""
mutable struct Sample{T}
    value::T
    logprob::Real
end


"""
    GrayBox.Environment

Alias type for a dictionary of gray-box environment distributions.

     e.g., `Environment(:variable_name => Sampleable)`
"""
const Environment = Dict{Symbol, Sampleable}


"""
    GrayBox.EnvironmentSample

Alias type for a single environment sample.

    e.g., `EnvironmentSample(:variable_name => Sample(value, logprob))`
"""
const EnvironmentSample = Dict{Symbol, Sample}


"""
    environment(sim::GrayBox.Simulation)

Return all distributions used in the simulation environment.
"""
function environment(sim::Simulation)::Environment end


"""
    transition!(sim::Union{GrayBox.Simulation, GrayBox.Simulation})::Real

Given an input `sample::EnvironmentSample`, apply transition and return the transition log-probability.
"""
function transition!(sim::Simulation, sample::EnvironmentSample)::Real end


"""
    transition!(sim::GrayBox.Simulation)::Real

Apply a transition step, and return the transition log-probability (used with `ASTSeedAction`).
"""
function transition!(sim::Simulation)::Real end


end # module GrayBox