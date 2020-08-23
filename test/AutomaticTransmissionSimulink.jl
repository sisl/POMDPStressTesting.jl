using Revise
using POMDPStressTesting
using Distributions
using Parameters
using MATLAB

# Automatic transmission Simulink model used
# in the 2019 ARCH-COMP falsification category.

@with_kw mutable struct ATParams
    model_name::String = "Autotrans_shift" # Simulink model name
    model::String = joinpath(@__DIR__, "models", model_name * ".mdl") # Simulink model full path
    time_range::Vector = [0, 30] # Simulation time
    throttle_range::Vector = [0, 100] # Throttle open percentage
    brake_range::Vector = [0, 325] # Brake torque
    max_actions::Int = 4 # Maximum number of actions
    speed_thresh::Real = 120
    stl::Function = yout->any(yout[:,1] .>= speed_thresh)
end


# Implement abstract BlackBox.Simulation
@with_kw mutable struct ATSim <: BlackBox.Simulation
    params::ATParams = ATParams() # Parameters
    action_matrix::Matrix{Float64} = Matrix{Float64}(undef, 0, 3) # Set of actions (note, Float64 not Real to allow MATLAB to convert)
    t::Real = 0 # Current time
    tout::Array = []
    yout::Array = []
end


# Override from BlackBox
function BlackBox.initialize!(sim::ATSim)
    sim.t = 0 # Reset current simulation time
    sim.action_matrix = Matrix{Float64}(undef, 0, 3) # Reset action set
    sim.tout = []
    sim.yout = []

    startime, endtime = sim.params.time_range
    times = startime:endtime
    @mput endtime # pass to MATLAB
    @mput times # pass to MATLAB

    # Load Simulink model in MATLAB
    eval_string("load_system('$(sim.params.model)')")
end


# Override from BlackBox
function BlackBox.transition!(sim::ATSim)
    # New time from "remaining" time (i.e. starting at sim.t)
    dist_time::Uniform = Uniform(sim.t, sim.params.time_range[end])
    dist_throttle::Uniform = Uniform(sim.params.throttle_range...)
    dist_brake::Uniform = Uniform(sim.params.brake_range...)

    sampled_time = rand(dist_time)
    sampled_throttle = rand(dist_throttle)
    sampled_brake = rand(dist_brake)
    sim.t = sampled_time # Keep track of latest time.

    input = [sampled_time sampled_throttle sampled_brake] # New input action.
    sim.action_matrix = [sim.action_matrix; input] # Append to action set.

    logprob::Real = logpdf(dist_time, sampled_time) +
                    logpdf(dist_throttle, sampled_throttle) +
                    logpdf(dist_brake, sampled_brake)
    return logprob::Real
end


# Override from BlackBox
BlackBox.distance!(sim::ATSim) = max(minimum(sim.params.speed_thresh .- sim.yout[:,1]), 0) # Non-negative


# Override from BlackBox
BlackBox.isevent!(sim::ATSim) = sim.params.stl(sim.yout)


# Override from BlackBox
BlackBox.isterminal!(sim::ATSim) = BlackBox.isevent!(sim) || size(sim.action_matrix, 1) >= sim.params.max_actions


# Override from BlackBox
function BlackBox.evaluate!(sim::ATSim)
    logprob::Real  = BlackBox.transition!(sim) # Step simulation

    x = sim.action_matrix # Matrix (not array of arrays)
    @mput x # Pass to MATLAB

    # `evalc` to suppress "data type override" print-out.
    eval_string("[~,results] = evalc('sim(''$(sim.params.model_name)'', ''StopTime'', num2str(endtime), ''LoadExternalInput'', ''on'', ''ExternalInput'', mat2str(x), ''SaveTime'', ''on'', ''TimeSaveName'', ''tout'', ''SaveOutput'', ''on'', ''OutputSaveName'', ''yout'', ''SaveFormat'', ''Array'', ''SolverType'', ''variable-step'', ''OutputOption'', ''SpecifiedOutputTimes'', ''ZeroCrossAlgorithm'', ''Adaptive'', ''OutputTimes'', mat2str(times))');")

    # MATLAB.jl cannot directly handle nested structs
    sim.tout = mat"results.tout"
    sim.yout = mat"results.yout"

    distance::Real = BlackBox.distance!(sim) # Calculate miss distance
    event::Bool    = BlackBox.isevent!(sim) # Check event indication
    if event
        @info "Failure event found mid-simulation!"
        header = [:time :speed :rpms :gears]
        display(vcat(header, [sim.tout sim.yout])) # TODO. add stl (:failure) category
    end

    return (logprob::Real, distance::Real, event::Bool)
end


function setup_ast()
    # Create black-box simulation object
    sim::BlackBox.Simulation = ATSim()

    # AST MDP formulation object
    mdp::ASTMDP = ASTMDP(sim)
    mdp.params.debug = true # record metrics
    mdp.params.seed = 1
    mdp.params.top_k = 10

    # Hyperparameters for MCTS-PW as the solver
    solver = MCTSASTSolver(depth=sim.params.time_range[end],
                           exploration_constant=10.0,
                           k_action=0.1,
                           alpha_action=0.85,
                           n_iterations=20)

    policy = solve(solver, mdp)

    return (policy, mdp, sim)
end


(policy, mdp, sim) = setup_ast()

action_trace = playout(mdp, policy)

final_state = playback(mdp, action_trace)

print_metrics(mdp)

nothing # Suppress REPL