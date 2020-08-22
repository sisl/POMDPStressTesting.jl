Base.hash(a::ASTAction) = hash(a.seed)
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

function Base.hash(s::ASTState)
    h = hash(s.t_index)
    h = hash(h, hash(isnothing(s.parent) ? nothing : s.parent.hash))
    h = hash(h, hash(s.action))
    return h
end

Base.:(==)(w::ASTAction,v::ASTAction) = w.seed == v.seed
Base.:(==)(w::ASTState,v::ASTState) = hash(w) == hash(v)
Base.isequal(w::ASTAction,v::ASTAction) = isequal(w.seed, v.seed)
Base.isequal(w::ASTState,v::ASTState) = hash(w) == hash(v)