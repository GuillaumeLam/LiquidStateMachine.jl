using LiquidStateMachine
using Test

@testset "LiquidStateMachine.jl" begin
    include("spike_train_tests.jl")
    include("lsm_tests.jl")
    include("util_tests.jl")
end
