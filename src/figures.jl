using PyPlot
using Seaborn # for kernel density
using Statistics

import .AST: ASTMDP, ASTMetrics

"""
Stacked figure with metrics over episodes:
    - Running miss distance mean
    - Minimum miss distance
    - Cumulative number of failure events
"""
episodic_figures(mdp::ASTMDP; kwargs...) = episodic_figures(mdp.metrics; kwargs...)
function episodic_figures(metrics::ASTMetrics; gui=false)
    miss_distances = metrics.miss_distance
    max_iters = length(miss_distances)

    PyPlot.pygui(gui) # Plot with GUI window (if true)
    fig = figure(figsize=(7,7))

    handles = []

    # Font size changes
    plt.rc("axes", titlesize=15, labelsize=13)
    plt.rc("legend", fontsize=12)


    ## Plot 1: Running miss distance mean
    ax = fig.add_subplot(3,1,1)
    title("Running Miss Distance Mean")

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0)

    ax.plot([mean(miss_distances[1:i]) for i in 1:max_iters], color="darkcyan", zorder=2)
    miss_std_below = [mean(miss_distances[1:i])-std(miss_distances[1:i]) for i in 1:max_iters]
    miss_std_above = [mean(miss_distances[1:i])+std(miss_distances[1:i]) for i in 1:max_iters]
    ax.fill_between(1:max_iters, miss_std_below, miss_std_above, color="darkcyan", alpha=0.1)

    ylabel("Miss Distance")
    ax.tick_params(labelbottom=false)

    ## Plot 2: Minimum miss distance
    ax = fig.add_subplot(3,1,2)
    title("Minimum Miss Distance")
    pl0 = ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0)
    pl1 = ax.plot([minimum(miss_distances[1:i]) for i in 1:max_iters], color="darkcyan", label=L"{AST}_{MCTS}")
    ylabel("Miss Distance")
    handles = [pl0, pl1[1]]

    ax.tick_params(labelbottom=false)


    ## Plot 3: Cumulative failures
    ax = fig.add_subplot(3,1,3)
    E = metrics.miss_distance .<= 0
    max_iters = length(E)

    title("Cumulative Number of Failure Events")
    ax.plot(cumsum(E[1:max_iters]), color="darkcyan")
    xlabel("Episode")
    ylabel("Number of Events")

    yscale("log")

    fig.legend(handles, ["Event Horizon", L"{\rm AST}_{\rm MCTS}"],
               columnspacing=0.8, loc="lower center", bbox_to_anchor=(0.52, 0), fancybox=true, shadow=false, ncol=5)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.13) # <-- Change the 0.02 to work for your plot.


    println("First failure: ", findfirst(E), "/", length(E))
    println("Number of failures: ", sum(E))
    println("Failure rate: ", sum(E)/length(E))
end


"""
Stacked figure with distributions:
    - Miss distance distribution
    - Log-likelihood distribution
"""
distribution_figures(mdp::ASTMDP; kwargs...) = distribution_figures(mdp.metrics; kwargs...)
function distribution_figures(metrics; gui=false)
    PyPlot.pygui(gui) # Plot with GUI window (if true)

    fig = figure(figsize=(6,5))

    # Font size changes
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("legend", fontsize=11)


    ## Plot 1: Miss distanace distribution
    subplot(2,1,1)
    n_bins = 20

    Seaborn.kdeplot(-metrics.miss_distance, bw=0.1, color="darkcyan", shade=true, cut=2000)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.0)
    xl = plt.xlim()
    plt.xlim([xl[1], xl[2]])

    legend(["Event Horizon", L"{\rm AST}_{\rm MCTS}"], ncol=2, columnspacing=0.5, numpoints=2)
    title("Miss Distance Distribution")
    xlabel(L"-d")
    ylabel("Density");
    yl=ylim()
    ylim([yl[1], yl[2]+0.001]) # y-buffer for legend fit and placement.


    ## Plot 2: Log-likelihood distribution
    subplot(2,1,2)
    Seaborn.kdeplot(metrics.logprob[findall(metrics.miss_distance .<= 0)], bw=0.1, color="darkcyan", shade=true, cut=100)
    legend([L"{\rm AST}_{\rm MCTS}"], loc="upper left")
    title("Log-Likelihood Distribution: Failure Events")
    xlabel(L"\log p")
    ylabel("Density");
    fig.tight_layout()
end