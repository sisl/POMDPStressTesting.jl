using Pkg

if VERSION < v"1.1"
    pkg"dev https://github.com/JuliaPOMDP/POMDPs.jl"
    pkg"dev https://github.com/JuliaPOMDP/RLInterface.jl"
else
    pkg"add https://github.com/JuliaPOMDP/RLInterface.jl"
end
