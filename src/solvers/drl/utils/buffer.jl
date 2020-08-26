# Modified from Shreyas Kowshik's implementation.

"""
Data dictionary buffer
"""
@with_kw mutable struct Buffer
    exp_dict::Dict = Dict()
end

register!(b::Buffer, name::String) = b.exp_dict[name] = []

"""
Add a variable for it's history to be logged
"""
add!(b::Buffer, name::String, value::Any) = push!(b.exp_dict[name], value)

import Base.get
get(b::Buffer, name::String) = b.exp_dict[name]

function clear!(b::Buffer, name=nothing)
	if name == nothing
		for key in keys(b.exp_dict)
			register!(b, key)
		end
	else
		@assert typeof(name) <: String "Name of key must be a string"
		register!(b, name)
	end
	return nothing
end


function initialize_episode_buffer()
    eb = Buffer()
    register!(eb,"states")
    register!(eb,"actions")
    register!(eb,"rewards")
    register!(eb,"next_states")
    register!(eb,"dones")
    register!(eb,"returns")
    register!(eb,"advantages")
    register!(eb,"log_probs")
    register!(eb,"kl_params")
    return eb
end


function initialize_stats_buffer()
    sb = Buffer()
    register!(sb,"rollout_rewards")
    return sb
end