# Library/Interface

This section details the interface and functions provided by POMDPStressTesting.jl.

```@meta
CurrentModule = POMDPStressTesting
```

## Contents

```@contents
Pages = ["interface.md"]
```

## Index

```@index
Pages = ["interface.md"]
```

# Modules
```@docs
POMDPStressTesting
```

```@docs
AST
```

```@docs
GrayBox
```

```@docs
BlackBox
```

## [`GrayBox`](@id graybox_interface)
```@docs
GrayBox.Simulation
GrayBox.Sample
GrayBox.Environment
GrayBox.EnvironmentSample
GrayBox.environment
GrayBox.transition!
```

## [`BlackBox`](@id blackbox_interface)
```@docs
BlackBox.initialize!
BlackBox.evaluate!
BlackBox.distance
BlackBox.isevent
BlackBox.isterminal
```

```@meta
CurrentModule = AST
```

## `AST`
```@docs
search!
initialstate
reward
gen
isterminal
discount
random_action
action
actions
convert_s
go_to_state
record
record_trace
get_top_path
rollout
rollout_end
playback
online_path
state_trace
action_trace
q_trace
action_q_trace
reset_metrics!
```

# AST Types
```@docs
ASTParams
ASTAction
ASTSeedAction
ASTSampleAction
ASTState
ASTMetrics
ASTMDP
```

```@meta
CurrentModule = POMDPStressTesting
```

# [Metrics](@id metrics)
```@docs
print_metrics
```

# [Visualizations/Figures](@id visualizations)
```@docs
visualize
episodic_figures
distribution_figures
```