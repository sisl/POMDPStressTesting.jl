import .AST: ASTMDP, ASTMetrics
using PyPlot
using Seaborn

"""
Stacked figure with metrics over episodes:

    - Running miss distance mean
    - Minimum miss distance
    - Cumulative number of failure events
"""
episodic_figures(mdp::ASTMDP; kwargs...) = episodic_figures(mdp.metrics; kwargs...)
function episodic_figures(metrics::ASTMetrics; gui::Bool=true, fillstd::Bool=false)
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

    rolling_mean = []
    d_sum = 0
    for i in 1:max_iters
        d_sum += miss_distances[i]
        push!(rolling_mean, d_sum/i)
    end
    # [mean(miss_distances[1:i]) for i in 1:max_iters]
    ax.plot(rolling_mean, color="darkcyan", zorder=2)
    if fillstd
        miss_std_below = [mean(miss_distances[1:i])-std(miss_distances[1:i]) for i in 1:max_iters]
        miss_std_above = [mean(miss_distances[1:i])+std(miss_distances[1:i]) for i in 1:max_iters]
        ax.fill_between(1:max_iters, miss_std_below, miss_std_above, color="darkcyan", alpha=0.1)
    end

    ylabel("Miss Distance")
    ax.tick_params(labelbottom=false)

    ## Plot 2: Minimum miss distance
    ax = fig.add_subplot(3,1,2)
    title("Minimum Miss Distance")
    pl0 = ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0)
    rolling_min = []
    current_min = Inf
    for i in 1:max_iters
        if miss_distances[i] < current_min
            current_min = miss_distances[i]
        end
        push!(rolling_min, current_min)
    end
    pl1 = ax.plot(rolling_min, color="darkcyan", label="AST")
    ylabel("Miss Distance")
    handles = [pl0, pl1[1]]

    ax.tick_params(labelbottom=false)


    ## Plot 3: Cumulative failures
    ax = fig.add_subplot(3,1,3)
    E = metrics.event
    max_iters = length(E)

    title("Cumulative Number of Failure Events")
    ax.plot(cumsum(E[1:max_iters]), color="darkcyan")
    xlabel("Episode")
    ylabel("Number of Events")

    yscale("log")

    fig.legend(handles, ["Event Horizon", "AST"],
               columnspacing=0.8, loc="lower center", bbox_to_anchor=(0.52, 0), fancybox=true, shadow=false, ncol=5)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.13) # <-- Change the 0.02 to work for your plot.

    return gcf()
end



"""
Stacked figure with distributions:

    - Miss distance distribution
    - Log-likelihood distribution
"""
distribution_figures(mdp::ASTMDP; kwargs...) = distribution_figures(mdp.metrics; kwargs...)
distribution_figures!(mdp::ASTMDP; kwargs...) = distribution_figures(mdp.metrics; kwargs..., hold=true)
distribution_figures!(metrics; kwargs...) = distribution_figures(metrics; kwargs..., hold=true)
function distribution_figures(metrics; color="darkcyan", gui=true, logprob=true, bw=[0.1, 0.1], only_failure=false, hold=false, label="AST")
    PyPlot.pygui(gui) # Plot with GUI window (if true)

    if hold
        fig = gcf()
    else
        fig = figure(figsize=(6,5))
    end

    # Font size changes
    plt.rc("axes", titlesize=12, labelsize=11)
    plt.rc("legend", fontsize=11)

    ## Plot 1: Miss distanace distribution
    subplot(2,1,1)

    Seaborn.kdeplot(metrics.miss_distance, bw=bw[1], color=color, shade=true, cut=200, label=label) # cut=2000
    hold == false && plt.axvline(x=0, color="black", linestyle="--", linewidth=1.0, label="Event Horizon")
    xl = plt.xlim()
    plt.xlim([xl[1], xl[2]])

    # legend(["Event Horizon", "AST"], ncol=2, columnspacing=0.5, numpoints=2)
    legend(ncol=1, columnspacing=0.5, numpoints=2)
    title("Miss Distance Distribution")
    xlabel(L"d")
    ylabel("Density");
    yl=ylim()
    ylim([yl[1], yl[2]+0.001]) # y-buffer for legend fit and placement.

    ## Plot 2: (Log-)likelihood distribution
    subplot(2,1,2)
    if logprob
        p = metrics.logprob[findall(metrics.event)]
        p_title = "Log-likelihood"
        p_xlabel = L"\log p"
    else
        p = exp.(metrics.logprob[findall(metrics.event)])
        p_title = "Likelihood"
        p_xlabel = L"p"
    end
    Seaborn.kdeplot(p, bw=bw[2], color=color, shade=true, cut=100, label=label)
    # legend(["AST"], loc="upper left")
    legend(loc="upper left")
    title("$p_title Distribution: Failure Events")
    xlabel(p_xlabel)
    ylabel("Density");
    if logprob == false
        # restrict x-axis to [0,1] for probability
        xlim([0, 1])
    end
    fig.tight_layout()

    return gcf()
end