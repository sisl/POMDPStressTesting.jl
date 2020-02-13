"""
Handles generating random seeds and associated helper functions.
"""
module RandomSeedGenerator

# Credit: Ritchie Lee

export RSG, next!, set_global_seed

using Random
using Base.Iterators
using IterTools

struct RSG
    state::Vector{UInt32}
end
RSG(len::Int64=1, seed::Int64=0) = seed_to_state_itr(len, seed) |> collect |> RSG

# TODO: Clean up.
# set_from_seed!(rsg::RSG, len::Int64, seed::Int64) = copy!(rsg.state, seed_to_state_itr(len, seed))
seed_to_state_itr(len::Int64, seed::Int64) = take(iterated(hash_uint32, seed), len)
hash_uint32(x) = UInt32(hash(x) & 0x00000000FFFFFFFF) # Take lower 32-bits

function next!(rsg::RSG)
	map!(hash_uint32, rsg.state, rsg.state)
	return rsg::RSG
end

function next(rsg::RSG)
	rsg′ = deepcopy(rsg)
	next!(rsg′)
	return rsg′
end

import Random.seed!
set_global_seed(rsg::RSG) = set_gv_rng_state(rsg.state)
set_gv_rng_state(i::UInt32) = set_gv_rng_state([i])
set_gv_rng_state(a::Vector{UInt32}) = Random.seed!(a)

Base.length(rsg::RSG) = length(rsg.state)
Base.hash(rsg::RSG) = hash(rsg.state)
Base.:(==)(rsg1::RSG, rsg2::RSG) = rsg1.state == rsg2.state
Base.isequal(rsg1::RSG, rsg2::RSG) = rsg1 == rsg2

end # module RandomSeedGenerator