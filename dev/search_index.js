var documenterSearchIndex = {"docs":
[{"location":"guide/#guide","page":"Guide","title":"Guide","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"To use this package to stress test your own system, the user has to provide the following:","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Implementations of the BlackBox system interface functions (outlined here)\nImplementations of the GrayBox simulator/environment interface functions (outlined here)","category":"page"},{"location":"guide/#Problem-Setup","page":"Guide","title":"Problem Setup","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Once the GrayBox and BlackBox interfaces are defined (see Example for a full example), the user has access to the solvers and simply needs to set up the AST problem.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"First, set up your simulation structure, where YourSimulation <: GrayBox.Simulation (note, you'll need to replace YourSimulation with your own structure you've defined as part of the GrayBox interface):","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"sim::GrayBox.Simulator = YourSimulation()","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"Then, set up the adaptive stress testing (AST) Markov decision process (MDP) structure, given your sim object:","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"mdp::ASTMDP = ASTMDP{ASTSeedAction}(sim) # could use `ASTSeedAction` or `ASTSampleAction`","category":"page"},{"location":"guide/#ast_action_type","page":"Guide","title":"AST Action Type","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"ASTSeedAction\nThis action type samples seeds for a random number generator (RNG), which means the GrayBox.transition! function must sample from the environment themselves and apply the transition.\nUseful when it's difficult to access the individual sampled environment outputs.\nASTSampleAction\nThis action type samples directory from the GrayBox.Environment and will pass the sample(s) (as GrayBox.EnvironmentSample) to the GrayBox.transition! function to be directly applied\nUseful when you have full access to the simulation environment and can apply each sample directly in the transition function.","category":"page"},{"location":"guide/#Solving-the-AST-MDP","page":"Guide","title":"Solving the AST MDP","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Now you can choose your solver (see Solvers) and run solve given the AST mdp (Markov decision process) to produce an online planner (no search has been performed yet):","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"solver = MCTSPWSolver()\nplanner = solve(solver, mdp)","category":"page"},{"location":"guide/#Searching-for-Failures","page":"Guide","title":"Searching for Failures","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Once the problem is set up, you can search for failures using search! given your planner. This will return the best action trace it found.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"action_trace = search!(planner)","category":"page"},{"location":"guide/#Playback-Failures","page":"Guide","title":"Playback Failures","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Given either the action_trace or the top k performing action traces, using get_top_path(mdp), you can playback the particular action:","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"final_state = playback(planner, action_trace)","category":"page"},{"location":"guide/#metrics_visualizations","page":"Guide","title":"Metrics and Visualizations","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"Afterwards, you can look at performance metrics and visualizations (see Metrics and Visualization):","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"print_metrics(planner)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"To plot, first install the PyPlot and Seaborn packages and load them. We use Requires.jl to handle these dependencies.","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"using PyPlot\nusing Seaborn","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"You can plot episodic metrics, including running miss distance mean, minimum miss distance, and cumulative failures all over episode (i.e. iteration):","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"episodic_figures(planner)","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"You can also plot the miss distance distribution and log-likelihood distribution:","category":"page"},{"location":"guide/","page":"Guide","title":"Guide","text":"distribution_figures(planner)","category":"page"},{"location":"interface/#Library/Interface","page":"Library/Interface","title":"Library/Interface","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"This section details the interface and functions provided by POMDPStressTesting.jl.","category":"page"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"CurrentModule = POMDPStressTesting","category":"page"},{"location":"interface/#Contents","page":"Library/Interface","title":"Contents","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"Pages = [\"interface.md\"]","category":"page"},{"location":"interface/#Index","page":"Library/Interface","title":"Index","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"Pages = [\"interface.md\"]","category":"page"},{"location":"interface/#Modules","page":"Library/Interface","title":"Modules","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"POMDPStressTesting","category":"page"},{"location":"interface/#POMDPStressTesting.POMDPStressTesting","page":"Library/Interface","title":"POMDPStressTesting.POMDPStressTesting","text":"Adaptive Stress Testing for the POMDPs.jl ecosystem.\n\n\n\n\n\n","category":"module"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"AST","category":"page"},{"location":"interface/#POMDPStressTesting.AST","page":"Library/Interface","title":"POMDPStressTesting.AST","text":"Provides implementation of Adaptive Stress Testing (AST) formulation of MDPs/POMDPs.\n\n\n\n\n\n","category":"module"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"GrayBox","category":"page"},{"location":"interface/#POMDPStressTesting.AST.GrayBox","page":"Library/Interface","title":"POMDPStressTesting.AST.GrayBox","text":"Provides virtual interface for the gray-box environment and simulator.\n\n\n\n\n\n","category":"module"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"BlackBox","category":"page"},{"location":"interface/#POMDPStressTesting.AST.BlackBox","page":"Library/Interface","title":"POMDPStressTesting.AST.BlackBox","text":"Provides virtual interface for the black-box system under test (SUT).\n\n\n\n\n\n","category":"module"},{"location":"interface/#graybox_interface","page":"Library/Interface","title":"GrayBox","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"GrayBox.Simulation\nGrayBox.Sample\nGrayBox.Environment\nGrayBox.EnvironmentSample\nGrayBox.environment\nGrayBox.transition!","category":"page"},{"location":"interface/#POMDPStressTesting.AST.GrayBox.Simulation","page":"Library/Interface","title":"POMDPStressTesting.AST.GrayBox.Simulation","text":"GrayBox.Simulation\n\nAbstract base type for a gray-box simulation.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.GrayBox.Sample","page":"Library/Interface","title":"POMDPStressTesting.AST.GrayBox.Sample","text":"GrayBox.Sample\n\nHolds sampled value and log-probability.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.GrayBox.Environment","page":"Library/Interface","title":"POMDPStressTesting.AST.GrayBox.Environment","text":"GrayBox.Environment\n\nAlias type for a dictionary of gray-box environment distributions.\n\n e.g., `Environment(:variable_name => Sampleable)`\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.GrayBox.EnvironmentSample","page":"Library/Interface","title":"POMDPStressTesting.AST.GrayBox.EnvironmentSample","text":"GrayBox.EnvironmentSample\n\nAlias type for a single environment sample.\n\ne.g., `EnvironmentSample(:variable_name => Sample(value, logprob))`\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.GrayBox.environment","page":"Library/Interface","title":"POMDPStressTesting.AST.GrayBox.environment","text":"environment(sim::GrayBox.Simulation)\n\nReturn all distributions used in the simulation environment.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.GrayBox.transition!","page":"Library/Interface","title":"POMDPStressTesting.AST.GrayBox.transition!","text":"transition!(sim::Union{GrayBox.Simulation, GrayBox.Simulation})::Real\n\nGiven an input sample::EnvironmentSample, apply transition and return the transition log-probability.\n\n\n\n\n\ntransition!(sim::GrayBox.Simulation)::Real\n\nApply a transition step, and return the transition log-probability (used with ASTSeedAction).\n\n\n\n\n\n","category":"function"},{"location":"interface/#blackbox_interface","page":"Library/Interface","title":"BlackBox","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"BlackBox.initialize!\nBlackBox.evaluate!\nBlackBox.distance\nBlackBox.isevent\nBlackBox.isterminal","category":"page"},{"location":"interface/#POMDPStressTesting.AST.BlackBox.initialize!","page":"Library/Interface","title":"POMDPStressTesting.AST.BlackBox.initialize!","text":"initialize!(sim::GrayBox.Simulation)\n\nReset state to its initial state.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.BlackBox.evaluate!","page":"Library/Interface","title":"POMDPStressTesting.AST.BlackBox.evaluate!","text":"evaluate!(sim::GrayBox.Simulation)::Tuple{logprob::Real, miss_distance::Real, isevent::Bool}\nevaluate!(sim::GrayBox.Simulation, sample::GrayBox.EnvironmentSample)::Tuple{logprob::Real, miss_distance::Real, isevent::Bool}\n\nEvaluate the SUT given some input seed and current state, returns logprob, miss_distance, and isevent indication. If the sample version is implemented, then ASTSampleAction will be used instead of ASTSeedAction.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.BlackBox.distance","page":"Library/Interface","title":"POMDPStressTesting.AST.BlackBox.distance","text":"distance(sim::GrayBox.Simulation)::Real\n\nReturn how close to an event a terminal state was (i.e. some measure of \"miss distance\" to the event of interest).\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.BlackBox.isevent","page":"Library/Interface","title":"POMDPStressTesting.AST.BlackBox.isevent","text":"isevent(sim::GrayBox.Simulation)::Bool\n\nReturn a boolean indicating if the SUT reached a failure event of interest.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.BlackBox.isterminal","page":"Library/Interface","title":"POMDPStressTesting.AST.BlackBox.isterminal","text":"isterminal(sim::GrayBox.Simulation)::Bool\n\nReturn an indication that the simulation is in a terminal state.\n\n\n\n\n\n","category":"function"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"CurrentModule = AST","category":"page"},{"location":"interface/#AST","page":"Library/Interface","title":"AST","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"search!\ninitialstate\nreward\ngen\nisterminal\ndiscount\nrandom_action\naction\nactions\nconvert_s\ngo_to_state\nrecord\nrecord_trace\nget_top_path\nrollout\nrollout_end\nplayback\nonline_path\nstate_trace\naction_trace\nq_trace\naction_q_trace\nreset_metrics!","category":"page"},{"location":"interface/#POMDPStressTesting.AST.search!","page":"Library/Interface","title":"POMDPStressTesting.AST.search!","text":"search!(planner)\n\nSearch for failures given a planner. Implemented by each solver.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPs.initialstate","page":"Library/Interface","title":"POMDPs.initialstate","text":"Initialize AST MDP state. Overridden from POMDPs.initialstate interface.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPs.reward","page":"Library/Interface","title":"POMDPs.reward","text":"reward(mdp::ASTMDP, logprob::Real, isevent::Bool, isterminal::Bool, miss_distance::Real)::Float64\n\nReward function for the AST formulation. Defaults to:\n\nR_E       if isterminal and isevent (1)\n-d        if isterminal and !isevent (2)\nlog(p)    otherwise (3)\n\n1) Terminates with event, collect reward_bonus (defaults to 0)\n2) Terminates without event, collect negative miss distance\n3) Each non-terminal step, accumulate reward correlated with the transition probability\n\nFor epsidic reward problems (i.e. rewards only at the end of an episode), set mdp.params.episodic_rewards to get:\n\n(log(p) - d)*R_E    if isterminal and isevent (1)\nlog(p) - d          if isterminal and !isevent (2)\n0                   otherwise (3)\n\n1) Terminates with event, collect transition probability and miss distance with multiplicative reward bonus\n2) Terminates without event, collect transitions probability and miss distance\n3) Each non-terminal step, no intermediate reward (set `mdp.params.give_intermediate_reward` to use log transition probability)\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPs.gen","page":"Library/Interface","title":"POMDPs.gen","text":"Generate next state and reward for AST MDP (handles episodic reward problems). Overridden from POMDPs.gen interface.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPs.isterminal","page":"Library/Interface","title":"POMDPs.isterminal","text":"Determine if AST MDP is in a terminal state. Overridden from POMDPs.isterminal interface.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPs.discount","page":"Library/Interface","title":"POMDPs.discount","text":"AST problems are (generally) undiscounted to treat future reward equally. Overridden from POMDPs.discount interface.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.random_action","page":"Library/Interface","title":"POMDPStressTesting.AST.random_action","text":"Randomly select next action, independent of the state.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPs.action","page":"Library/Interface","title":"POMDPs.action","text":"Randomly select next action, independent of the state. Overridden from POMDPs.action interface.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPs.convert_s","page":"Library/Interface","title":"POMDPs.convert_s","text":"Used by the CommonRLInterface to interact with deep RL solvers.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.go_to_state","page":"Library/Interface","title":"POMDPStressTesting.AST.go_to_state","text":"Reset AST simulation to a given state; used by the MCTS DPWSolver as the reset_callback function.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.record","page":"Library/Interface","title":"POMDPStressTesting.AST.record","text":"AST.record(::ASTMDP, sym::Symbol, val)\n\nRecard an ASTMetric specified by sym.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.record_trace","page":"Library/Interface","title":"POMDPStressTesting.AST.record_trace","text":"Record the best paths from termination leaf node.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.get_top_path","page":"Library/Interface","title":"POMDPStressTesting.AST.get_top_path","text":"Get k-th top path from the recorded top_paths.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.rollout","page":"Library/Interface","title":"POMDPStressTesting.AST.rollout","text":"Rollout simulation for MCTS; used by the MCTS DPWSolver as the estimate_value function. Custom rollout records action trace once the depth has been reached.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.rollout_end","page":"Library/Interface","title":"POMDPStressTesting.AST.rollout_end","text":"Rollout to only execute SUT at end (p accounts for probabilities generated outside the rollout)\n\nUser defined:\n\nfeed_gen Function to feed best action, replaces call to gen when feeding\n\nfeed_type Indicate when to feed best action. Either at the start of the rollout :start, or mid-rollout :mid\n\nbest_callback Callback function to record best miss distance or reward for later feeding during rollout\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.playback","page":"Library/Interface","title":"POMDPStressTesting.AST.playback","text":"Play back a given action trace from the initialstate of the MDP.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.online_path","page":"Library/Interface","title":"POMDPStressTesting.AST.online_path","text":"Follow MCTS optimal path online calling action after each selected state.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.state_trace","page":"Library/Interface","title":"POMDPStressTesting.AST.state_trace","text":"Trace up the tree to get all ancestor states.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.action_trace","page":"Library/Interface","title":"POMDPStressTesting.AST.action_trace","text":"Trace up the tree to get all ancestor actions.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.q_trace","page":"Library/Interface","title":"POMDPStressTesting.AST.q_trace","text":"Trace up the tree and accumulate Q-values.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.action_q_trace","page":"Library/Interface","title":"POMDPStressTesting.AST.action_q_trace","text":"Trace up the tree to get all ancestor actions and summed Q-values.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.AST.reset_metrics!","page":"Library/Interface","title":"POMDPStressTesting.AST.reset_metrics!","text":"Clear data stored in mdp.metrics.\n\n\n\n\n\n","category":"function"},{"location":"interface/#AST-Types","page":"Library/Interface","title":"AST Types","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"ASTParams\nASTAction\nASTSeedAction\nASTSampleAction\nASTState\nASTMetrics\nASTMDP","category":"page"},{"location":"interface/#POMDPStressTesting.AST.ASTParams","page":"Library/Interface","title":"POMDPStressTesting.AST.ASTParams","text":"AST.ASTParams\n\nAdaptive Stress Testing specific simulation parameters.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.ASTAction","page":"Library/Interface","title":"POMDPStressTesting.AST.ASTAction","text":"Abstract type for the AST action variants.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.ASTSeedAction","page":"Library/Interface","title":"POMDPStressTesting.AST.ASTSeedAction","text":"AST.ASTSeedAction\n\nRandom seed AST action.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.ASTSampleAction","page":"Library/Interface","title":"POMDPStressTesting.AST.ASTSampleAction","text":"AST.ASTSampleAction\n\nRandom environment sample as the AST action.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.ASTState","page":"Library/Interface","title":"POMDPStressTesting.AST.ASTState","text":"AST.ASTState\n\nState of the AST MDP.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.ASTMetrics","page":"Library/Interface","title":"POMDPStressTesting.AST.ASTMetrics","text":"AST.ASTMetrics\n\nDebugging metrics.\n\n\n\n\n\n","category":"type"},{"location":"interface/#POMDPStressTesting.AST.ASTMDP","page":"Library/Interface","title":"POMDPStressTesting.AST.ASTMDP","text":"AST.ASTMDP\n\nAdaptive Stress Testing MDP problem formulation object.\n\n\n\n\n\n","category":"type"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"CurrentModule = POMDPStressTesting","category":"page"},{"location":"interface/#metrics","page":"Library/Interface","title":"Metrics","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"print_metrics","category":"page"},{"location":"interface/#POMDPStressTesting.print_metrics","page":"Library/Interface","title":"POMDPStressTesting.print_metrics","text":"print_metrics(planner)\nprint_metrics(mdp::ASTMDP)\nprint_metrics(metrics::ASTMetrics)\n\nPrint failure metrics including:\n\niteration of first failure\nnumber of failures\nfailure rate\nhighest log-likelihood of failure\n\n\n\n\n\n","category":"function"},{"location":"interface/#visualizations","page":"Library/Interface","title":"Visualizations/Figures","text":"","category":"section"},{"location":"interface/","page":"Library/Interface","title":"Library/Interface","text":"visualize\nepisodic_figures\ndistribution_figures","category":"page"},{"location":"interface/#POMDPStressTesting.visualize","page":"Library/Interface","title":"POMDPStressTesting.visualize","text":"Visualize MCTS tree structure for AST MDP.\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.episodic_figures","page":"Library/Interface","title":"POMDPStressTesting.episodic_figures","text":"Stacked figure with metrics over episodes:\n\n- Running miss distance mean\n- Minimum miss distance\n- Cumulative number of failure events\n\n\n\n\n\n","category":"function"},{"location":"interface/#POMDPStressTesting.distribution_figures","page":"Library/Interface","title":"POMDPStressTesting.distribution_figures","text":"Stacked figure with distributions:\n\n- Miss distance distribution\n- Log-likelihood distribution\n\n\n\n\n\n","category":"function"},{"location":"solvers/#solvers","page":"Solvers","title":"Solvers","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"This section describes the solution methods (i.e. solvers) used to in the AST problem formulation to search for likely failure events.","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Several solvers are implemented:","category":"page"},{"location":"solvers/#Reinforcement-learning","page":"Solvers","title":"Reinforcement learning","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Monte Carlo tree search (MCTS) is used as the search algorithm in the standard AST formulation. We piggy-back on the MCTS.jl package with modification specific to AST. Modifications to MCTS are described here: [1]","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"MCTSPWSolver","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"[1] Robert J. Moss, Ritchie Lee, Nicholas Visser, Joachim Hochwarth, James G. Lopez, Mykel J. Kochenderfer, \"Adaptive Stress Testing of Trajectory Predictions in Flight Management Systems\", DASC 2020. https://arxiv.org/abs/2011.02559","category":"page"},{"location":"solvers/#Deep-reinforcement-learning","page":"Solvers","title":"Deep reinforcement learning","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Deep reinforcement learning solvers include trust region policy optimization (TRPO) [2] and proximal policy optimization (PPO) [3]. We'd like to thank Shreyas Kowshik's for their initial Julia implementation of these methods.","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"TRPOSolver\nPPOSolver","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"[2] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel, \"Trust Region Policy Optimization\", ICML 2015. https://arxiv.org/abs/1502.05477","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"[3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov, \"Proximal Policy Optimization\", 2017. https://arxiv.org/abs/1707.06347","category":"page"},{"location":"solvers/#Stochastic-optimization","page":"Solvers","title":"Stochastic optimization","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Solvers that use stochastic optimization include the cross-entropy method solver CEMSolver. This solution method is adapted from the CrossEntropyMethod.jl package by Anthony Corso.","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"CEMSolver","category":"page"},{"location":"solvers/#Baselines","page":"Solvers","title":"Baselines","text":"","category":"section"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"Baseline solvers are used for comparison to more sophisticated search methods. Currently, the only baseline solver is the RandomSearchSolver which uses Monte Carlo rollouts of a random policy to search for failures.","category":"page"},{"location":"solvers/","page":"Solvers","title":"Solvers","text":"RandomSearchSolver","category":"page"},{"location":"install/#Installation","page":"Installation","title":"Installation","text":"","category":"section"},{"location":"install/","page":"Installation","title":"Installation","text":"To install the package, run:","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"] add https://github.com/sisl/POMDPStressTesting.jl","category":"page"},{"location":"install/#Testing","page":"Installation","title":"Testing","text":"","category":"section"},{"location":"install/","page":"Installation","title":"Installation","text":"(Image: Build Status) (Image: codecov)","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"To run the test suite, open the Julia Pkg mode using ] and then run:","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"test POMDPStressTesting","category":"page"},{"location":"install/","page":"Installation","title":"Installation","text":"Testing is automated using Travis CI, which runs the test/runtests.jl file.","category":"page"},{"location":"contrib/#Contributing","page":"Contributing","title":"Contributing","text":"","category":"section"},{"location":"contrib/","page":"Contributing","title":"Contributing","text":"We welcome all contributions!","category":"page"},{"location":"contrib/","page":"Contributing","title":"Contributing","text":"Please fork the repository and submit a new Pull Request\nReport issues through our GitHub issue tracker\nFor further support, either file an issue or email Robert Moss at mossr@cs.stanford.edu","category":"page"},{"location":"contrib/#Style-Guide","page":"Contributing","title":"Style Guide","text":"","category":"section"},{"location":"contrib/","page":"Contributing","title":"Contributing","text":"(Image: Code Style: Blue)","category":"page"},{"location":"contrib/","page":"Contributing","title":"Contributing","text":"We follow the Blue style guide for Julia.","category":"page"},{"location":"#POMDPStressTesting","page":"Home","title":"POMDPStressTesting","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Build Status) (Image: codecov)","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package is used to find likely failures in a black-box software system. The package is integrated with the POMDPs.jl ecosystem; giving access to solvers, policies, and visualizations (although no prior knowledge of the POMDPs.jl package is needed to use POMDPStressTesting.jl—see the Guide). It uses a technique called adaptive stress testing (AST)[1] to find likely failures using a distance metric to a failure event and the likelihood of an environment sample to guide the search.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A POMDP is a partially observable Markov decision process, which is a framework to define a sequential decision making problem where the true state is unobservable. In the context of this package, we use the POMDP acronym mainly to tie the package to the POMDPs.jl package, but the system that is stress tested can also be defined as a POMDP.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package is intended to help developers stress test their systems before deployment into the real-world (see existing use cases for aircraft collision avoidance systems[1] and aircraft trajectory prediction systems[2]). It is also used for research purposes to expand on the AST concept by allowing additional solution methods to be explored and tested.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"[1] Ritchie Lee et al., \"Adaptive Stress Testing: Finding Likely Failure Events with Reinforcement Learning \", 2020. https://arxiv.org/abs/1811.02188","category":"page"},{"location":"","page":"Home","title":"Home","text":"[2] Robert J. Moss, Ritchie Lee, Nicholas Visser, Joachim Hochwarth, James G. Lopez, Mykel J. Kochenderfer, \"Adaptive Stress Testing of Trajectory Predictions in Flight Management Systems\", DASC 2020. https://arxiv.org/abs/2011.02559","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Package-Features","page":"Home","title":"Package Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Search for failures in a black-box system\nDefine probability distributions of your simulation environment\nFind likely system failures using a variety of solvers\nCalculate and visualize failure metrics\nReplay found failures","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\n    \"installation.md\",\n    \"guide.md\",\n    \"solvers.md\",\n    \"example.md\",\n    \"contrib.md\"\n]","category":"page"},{"location":"#system","page":"Home","title":"BlackBox System Definition","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A black-box system could be an external software executable, code written in Julia, or code written in another language. The system is generally a sequential decision making system than can be stepped foward in time. It is termed \"black-box\" because all we need is to be able to initialize it (using BlackBox.initialize!), evaluate or step the system forward in time (using BlackBox.evaluate!), and parse the output of the system to determine the distance metric (using BlackBox.distance), the failure event indication (using BlackBox.isevent), and whether the system is in a terminal state (using BlackBox.isterminal).","category":"page"},{"location":"","page":"Home","title":"Home","text":"See the BlackBox interface for implementation details","category":"page"},{"location":"","page":"Home","title":"Home","text":"The BlackBox system interface includes:","category":"page"},{"location":"","page":"Home","title":"Home","text":"BlackBox.initialize!(sim::Simulation) to initialize/reset the system under test\nBlackBox.evaluate!(sim::Simulation) to evaluate/execute the system under test\nBlackBox.distance(sim::Simulation) to return how close we are to an event\nBlackBox.isevent(sim::Simulation) to indicate if a failure event occurred\nBlackBox.isterminal(sim::Simulation) to indicate the simulation is in a terminal state","category":"page"},{"location":"#sim_env","page":"Home","title":"GrayBox Simulator/Environment Definition","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The gray-box simulator and environment define the parameters of your simulation and the probability distributions governing your simulation environment. It is termed \"gray-box\" because we need access to the probability distributions of the environment in order to get the log-likelihood of a sample used by the simulator (which is ulimately used by the black-box system).","category":"page"},{"location":"","page":"Home","title":"Home","text":"See the GrayBox interface for implementation details.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The GrayBox simulator and environment interface includes:","category":"page"},{"location":"","page":"Home","title":"Home","text":"GrayBox.Simulation type to hold simulation variables\nGrayBox.environment(sim::Simulation) to return the collection of environment distributions\nGrayBox.transition!(sim::Simulation) to transition the simulator, returning the log-likelihood","category":"page"},{"location":"#Failure-and-Distance-Definition","page":"Home","title":"Failure and Distance Definition","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A failure event of the system under test is defined be the user. The user defines the function BlackBox.isevent to return an boolean indicating a failure or not given the current state of the simulation. An example failure used in the context of AST would be a collision when stress testing autonomous vehicles or aircraft collision avoidance systems.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The real-valued distance metric is used to indicate \"how close are we to a failure?\" and is defined by the user in the BlackBox.distance function. This metric is used to guide the search process towards failures by receiving a signal of the distance to a failure. An example distance metric for the autonomous vehicle problem would be the distance between the autonomous vehicle and a pedestrian, where if a failure is a collision with a pedestrian then we'd like to minimize this distance metric to find failures.","category":"page"},{"location":"#Citation","page":"Home","title":"Citation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you use this package for research purposes, please cite the following:","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: status)","category":"page"},{"location":"","page":"Home","title":"Home","text":"@article{moss2021pomdpstresstesting,\n  title = {{POMDPStressTesting.jl}: Adaptive Stress Testing for Black-Box Systems},\n  author = {Robert J. Moss},\n  journal = {Journal of Open Source Software},\n  year = {2021},\n  volume = {6},\n  number = {60},\n  pages = {2749},\n  doi = {10.21105/joss.02749}\n}","category":"page"},{"location":"example/#example","page":"Example","title":"Example Problem","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"(Image: Example Notebook)","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"This section walks through a full example to illustrate how to use the POMDPStressTesting package. See the Jupyter notebook for the full implementation.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Some definitions to note for this example problem:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"System: a one-dimensional walking agent\nEnvironment: distribution of random walking actions, sampled from a standard normal distribution mathcal N(01)\nFailure event: agent walks outside of the ±10 region\nDistance metric: how close to the ±10 edge is the agent?","category":"page"},{"location":"example/#Problem-Description-(Walk1D)","page":"Example","title":"Problem Description (Walk1D)","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"This problem, called Walk1D, samples random walking distances from a standard normal distribution mathcal N(01) and defines failures as walking past a certain threshold (which is set to ±10 in this example). AST will either select the seed which deterministically controls the sampled value from the distribution (i.e. from the transition model) or will directly sample the provided environmental distributions. These action modes are determined by the seed-action or sample-action options (i.e. ASTSeedAction or ASTSampleAction, respectively) . AST will guide the simulation to failure events using a notion of distance to failure, while simultaneously trying to find the set of actions that maximizes the log-likelihood of the samples.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Refer to the notebook for the full implementation.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"For the non-notebook version, see the Julia file test/Walk1D.jl","category":"page"}]
}
