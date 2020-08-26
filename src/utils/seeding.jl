# Handles generating random seeds and associated helper functions.
import Random: seed!

hash_uint32(seed::Float64) = hash_uint32(hash(seed))
hash_uint32(seed::Union{UInt32, UInt64}) = UInt32(hash(seed) & 0x00000000FFFFFFFF)

function set_seed!(mdp::ASTMDP)
    # Use initial seed from mdp.params.seed
    seed = isnothing(mdp.current_seed) ? mdp.params.seed : mdp.current_seed
    mdp.current_seed = hash_uint32(seed)
end

set_global_seed(seed::UInt32) = Random.seed!(seed)
set_global_seed(a::ASTSeedAction) = set_global_seed(a.seed)
