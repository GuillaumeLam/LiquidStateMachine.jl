using LiquidStateMachine
using Test

@testset "LiquidStateMachine.jl" begin
    include("lsm_tests.jl")
    include("util_tests.jl")
end
