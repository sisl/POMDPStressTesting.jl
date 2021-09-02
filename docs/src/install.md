# Installation

To install the package, run:

```julia
] add https://github.com/sisl/POMDPStressTesting.jl
```

## Testing
[![Build Status](https://travis-ci.org/sisl/POMDPStressTesting.jl.svg?branch=master)](https://travis-ci.org/sisl/POMDPStressTesting.jl) [![codecov](https://codecov.io/gh/sisl/POMDPStressTesting.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/POMDPStressTesting.jl)


To run the test suite, open the Julia Pkg mode using `]` and then run:

```julia
test POMDPStressTesting
```

Testing is automated using Travis CI, which runs the [test/runtests.jl](https://github.com/mossr/POMDPStressTesting.jl/blob/master/test/runtests.jl) file.