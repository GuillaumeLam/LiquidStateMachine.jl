@testset "Positive" begin
    v = [1,-2,3]
    m = [v v]
    p = [1,0,0,2,3,0]

    @test LiquidStateMachine.genPositive(v) == p
    @test LiquidStateMachine.genPositive(m) == [p p]
end

@testset "Cap" begin
    v = [5,5,5]
    cap = [0,1,2]

    @test LiquidStateMachine.genCapped(v, cap) == cap
    @test LiquidStateMachine.genCapped([v v], cap) == [cap cap]
end

@testset "Discretize" begin
    v = [-3, 3, 1]
    c = [4,4,4]
    out1 = [4,0,4,0,4,0]
    out3 = [4,0,0,0,0,0,1.33333333,0,0,1.33333333,0,0]

    @test LiquidStateMachine.discretize(v,c,1) == out1
    @test isapprox(LiquidStateMachine.discretize(v,c,3), out3)
end
