using Pkg

if VERSION < v"1.1"
    pkg"dev https://github.com/JuliaPOMDP/POMDPs.jl"
end
pkg"add https://github.com/JuliaPOMDP/RLInterface.jl"
