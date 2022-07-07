# Base.hash(a::ASTSampleAction) = hash(a.???) # TODO.
Base.hash(a::ASTSeedAction) = hash(a.seed)
function Base.hash(A::Vector{ASTAction})
    if length(A) > 0
        h = hash(A[1])
        for i in 2:length(A)
            h = hash(h, hash(A[i]))
        end
    else
        h = hash([])
    end
    return h
end

Base.hash(s::ASTState, v::UInt=UInt(0)) = hash(s.t_index, s.parent, s.action, v)

function Base.hash(t_index::Int64, parent::Union{Nothing, ASTState}, action::Union{Nothing, ASTAction}, v::UInt=UInt(0))
    h = hash(t_index, v)
    h = hash((h, hash(isnothing(parent) ? nothing : parent.hash)), v)
    h = hash((h, hash(action)), v)
    return h
end

# Base.:(==)(w::ASTSampleAction,v::ASTSampleAction) = w.??? == v.??? # TODO.
Base.:(==)(w::ASTSeedAction,v::ASTSeedAction) = w.seed == v.seed
Base.:(==)(w::ASTState,v::ASTState) = hash(w) == hash(v)
# Base.isequal(w::ASTSampleAction,v::ASTSampleAction) = isequal(w.???, v.???) # TODO.
Base.isequal(w::ASTSeedAction,v::ASTSeedAction) = isequal(w.seed, v.seed)
Base.isequal(w::ASTState,v::ASTState) = hash(w) == hash(v)