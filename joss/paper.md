---
title: 'POMDPStressTesting.jl: Adaptive stress testing for black-box systems'
tags:
  - Julia
  - stress testing
  - black-box systems
  - POMDPs.jl
authors:
  - name: Robert J. Moss
    orcid: 0000-0003-2403-454X
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 26 August 2020
bibliography: paper.bib
header-includes: |
    \usepackage{listings}
---
\lstdefinelanguage{Julia}
  {keywords=[3]{initialize!, transition!, evaluate!, distance!, isevent!, isterminal!, environment},
   keywords=[2]{Nothing, Tuple, Real, Bool, Simulation, BlackBox, GrayBox, Sampleable, Environment},
   keywords=[1]{function, abstract, type, end},
   sensitive=true,
   morecomment=[l]\#,
   morecomment=[n]{\#=}{=\#},
   morestring=[s]{"}{"},
   morestring=[m]{'}{'},
   alsoletter=!?,
   literate={,}{{\color[HTML]{0F6FA3},}}1
            {\{}{{\color[HTML]{0F6FA3}\{}}1
            {\}}{{\color[HTML]{0F6FA3}\}}}1
}

# Summary
\href{https://github.com/sisl/POMDPStressTesting.jl}{POMDPStressTesting.jl} is a package that uses reinforcement learning and stochastic optimization to find likely failures in black-box systems through a technique called adaptive stress testing [@ast].
Adaptive stress testing (AST) has been used to find failures in safety-critical systems such as aircraft collision avoidance [@ast_acasx], flight management systems [@ast_fms], and autonomous vehicles [@ast_av].
The POMDPStressTesting.jl package is written in Julia [@julia] and is part of the wider POMDPs.jl ecosystem [@pomdps_jl].
Fitting into the POMDPs.jl ecosystem, our package has access to simulation tools, policies, visualizations, and---most importantly---solvers.
We provide several different solver variants including reinforcement learning solvers such as Monte Carlo tree search [@mcts] and deep reinforcement learning solvers such as trust region policy optimization (TRPO) [@trpo] and proximal policy optimization (PPO) [@ppo].
We also include stochastic optimization solvers such as the cross-entropy method [@cem] and include random search as a baseline.
Additional solvers can easily be added by adhering to the POMDPs.jl interface.

The AST formulation treats the falsification problem (i.e. finding failures) as a Markov decision process with a reward function that uses a notion of distance to a failure event to guide the search towards failure.
The reward function also uses the state-transition probabilities to guide towards \textit{likely} failures.
Recall that reinforcement learning aims to maximize the discounted sum of expected rewards, therefore using the log-likelihood allows us to maximize the summations, which is equivalent to maximizing the product of the likelihoods.
A gray-box simulation environment steps the simulation and outputs the state-transition probabilities, and the black-box system under test is evaluated in the simulator and outputs an event indication and the real-valued distance metric.
To apply AST to a general black-box system, a user has to implement the following interface:



The simulator stores simulation-specific parameters and the environment stores a collection of probability distributions that define the state-transitions (e.g., Gaussian noise models, uniform control inputs, etc.).
Two types of AST action modes are provided to the user: random seed actions or directly sampled actions.
The seed-action approach is useful when the user does not have direct access to the environmental distributions or when the environment is complex.
In this case, if the state-transition probability is inaccessible it may be set to $0$, thus guiding the search solely by the distance metric.
When using directly sampled actions, the \textsc{Transition} and \textsc{Evaluate} functions can take in an environment sample selected by the solvers and apply it directly as input to the black-box system, allowing for finer control over the search.

As an example, the functions in the above interface can either be implemented directly in Julia or can call out to C++, Python, MATLAB$^\text{\textregistered}$ or run a command line executable.
We provide a benchmark example often used in the falsification literature [@at_simulink] which uses a Simulink$^\text{\textregistered}$ automatic transmission model as the black-box system and selects throttle and brake control inputs as part of the environment.
Typically, implementing the \textsc{Distance} and \textsc{IsEvent} functions rely solely on the output of the black-box system under test, thus keeping in accordance with the black-box formulation.

Our package builds off work originally done in the AdaptiveStressTesting.jl package, but POMDPStressTesting.jl adheres to the interface defined by the POMDPs.jl package and provides different action modes and solver types.
Related falsification tools (i.e. tools that do not include most-likely failure analysis) are \textsc{S-TaLiRo} [@staliro], Breach [@breach], \textsc{Rrt-Rex} [@rrtrex], and \textsc{FalStar} [@falstar].
These packages use a combination of optimization, path planning, and reinforcement learning techniques to solve the falsification problem.
The tool closely related to POMDPStressTesting.jl is the AST Toolbox in Python [@ast_av], which wraps around the gym reinforcement learning environment [@gym].
The author has contributed to the AST Toolbox and found the need to create a similar package in Julia for better performance and to interface with the POMDPs.jl ecosystem in pure Julia.


# Research and Industrial Usage

POMDPStressTesting.jl has been used to find likely failures in aircraft trajectory prediction systems [@ast_fms], which are flight-critical subsystems used to aid in-flight automation.
A developmental commercial flight management system was stress tested so the system engineers could mitigate potential issues before system deployment.
In addition to traditional requirements-based testing for avionics certification [@do178c], this work is being used to find potential problems during development.
Other ongoing research is using POMDPStressTesting.jl for assessing the risk of autonomous vehicles.


# Acknowledgments

We acknowledge Ritchie Lee for his guidance and original work on the AdaptiveStressTesting.jl package and Mark Koren and Anthony Corso for their work on the AST Toolbox Python package and the CrossEntropyMethod.jl package.
We also acknowledge Shreyas Kowshik for his initial implementation of the TRPO and PPO algorithms in Julia.
We want to thank the Stanford Intelligent Systems Laboratory for their development of the POMDPs.jl ecosystem and the MCTS.jl package.
We also want to thank Mykel J. Kochenderfer for his support and research input and for advancing the Julia community.


# References