# Handles generating random seeds and associated helper functions.
import Random: seed!

hash_uint32(seed::UInt32) = UInt32(hash(seed) & 0x00000000FFFFFFFF)

function set_seed!(mdp::ASTMDP)
    mdp.current_seed = hash_uint32(mdp.current_seed)
end

set_global_seed(seed::UInt32) = Random.seed!(seed)
set_global_seed(a::ASTAction) = set_global_seed(a.seed)
