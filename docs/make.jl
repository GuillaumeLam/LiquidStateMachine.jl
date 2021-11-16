using Documenter, LiquidStateMachine

makedocs(;
    sitename="LiquidStateMachine.jl Documentation",
    pages = [
        "Home" => "index.md",
        "Examples" => "example.md",
        "Structure" => "structure.md",
        "Documentation" => "documentation.md",
        "References" => "references.md"
    ]
)

deploydocs(;
    repo="GuillaumeLam/LiquidStateMachine.jl.git",
    devbranch="main",
    push_preview=true
)
