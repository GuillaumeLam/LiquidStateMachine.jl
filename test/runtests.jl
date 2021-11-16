using LiquidStateMachine
using Test

@testset "LiquidStateMachine.jl" begin
    # A basic test
    params = LSM_Params(4,2)
    lsm = LSM(params)
    x = [1.0, 2.0, 5.0, 3.0]
    y = lsm(x)
    @test isa(y, Vector{Float64})
end
