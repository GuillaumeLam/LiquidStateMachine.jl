using Flux
using CUDA

@testset "LSM" begin
    @testset "Basic" begin
        # A basic test
        params = LSM_Params(4,2)
        lsm = LSM(params)
        x = [1.0, 2.0, 5.0, 3.0]
        y = lsm(x)

        println(typeof(y))
        @test isa(y, Vector{Float64})
        # @test isa(lsm([x x]), Vector{Float64})
    end

    @testset "Visual" begin
        # A basic test
        params = LSM_Params(4,2)
        lsm = LSM(params, visual=true)
        x = [1.0, 2.0, 5.0, 3.0]
        lsm(x)
        lsm(x)
        lsm(x)

        @test size(lsm.states_dict["spike"])[1] == 3
    end

    @testset "Overload" begin
        params = LSM_Params(4,2)
        lsm = LSM(params)

        @test Flux.trainable(lsm) == (lsm.readout,)
        @test CUDA.device(lsm) == Val(:cpu)
    end

    # @testset "Constructor" begin
    #
    # end
end
