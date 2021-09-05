# Installation

To install the package, run:

```julia
] add POMDPStressTesting
```

## Testing
[![Build Status](https://github.com/sisl/POMDPStressTesting.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/POMDPStressTesting.jl/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/sisl/POMDPStressTesting.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/POMDPStressTesting.jl)


To run the test suite, open the Julia Pkg mode using `]` and then run:

```julia
test POMDPStressTesting
```

Testing is automated using Travis CI, which runs the [test/runtests.jl](https://github.com/mossr/POMDPStressTesting.jl/blob/master/test/runtests.jl) file.