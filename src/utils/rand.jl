import Base.rand
function rand(rng::AbstractRNG, d::GrayBox.Environment)
    # Sample from each distribution in the dictionary
    # (similar to Anthoy Corso's CrossEntropyMethod.jl)
    sample = GrayBox.EnvironmentSample()
    for k in keys(d)
        value = rand(rng, d[k])
        logprob = logpdf(d[k], value)
        sample[k] = GrayBox.Sample(value, logprob)
    end
    return sample::GrayBox.EnvironmentSample
end


import Distributions.logpdf
function logpdf(sample::GrayBox.EnvironmentSample)
    sum(sample[k].logprob for k in keys(sample))
end
