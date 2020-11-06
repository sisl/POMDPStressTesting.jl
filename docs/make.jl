using Documenter, POMDPStressTesting

makedocs(
    modules = [POMDPStressTesting, AST, BlackBox, GrayBox],
    sitename = "POMDPStressTesting.jl",
    authors = "Robert Moss",
    clean = false,
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Installation" => "install.md",
            "Guide" => "guide.md",
            "Solvers" => "solvers.md",
        ],
        "Example" => "example.md",
        "Library/Interface" => "interface.md",
        "Contributing" => "contrib.md"
    ]
)

deploydocs(
    repo = "github.com/sisl/POMDPStressTesting.jl.git",
)
