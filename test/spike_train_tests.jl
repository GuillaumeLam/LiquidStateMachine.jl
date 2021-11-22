using Distributions

@testset "Spike Train" begin
    @testset "Generator" begin
        st_gen = SpikeTrainGenerator(Distributions.Bernoulli)
        n = 5
        x = zeros(n)
        st = st_gen(x)
        @test isa(st, Function)

        @test isa(st(0), AbstractVector)
        @test length(st(0)) == n
    end

    @testset "Generator Matrix" begin
        st_gen = SpikeTrainGenerator(Distributions.Bernoulli)
        n = 5
        x = zeros(n,n)
        st = st_gen(x)
        @test isa(st, AbstractMatrix)

        @test isa(st[1](0), AbstractVector)
        @test length(st[1](0)) == n
        @test length(st) == n
    end

    @testset "Decipher" begin
        st_dec = SpikeTrainDecipher(x->2*x)
        @test isa(st_dec.f,Function)

        st_dec = SpikeTrainDecipher()
        n = 5
        x = zeros(n)
        @test st_dec(x)==x
    end
end
