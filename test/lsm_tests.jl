using Flux
using CUDA

@testset "LSM" begin
    @testset "Basic" begin
        # A basic test
        params = LSM_Params(4,2)
        lsm = LSM(params)
        x = [1.0, 2.0, 5.0, 3.0]
        y = lsm(x)

        @test isa(y, Vector{Float64})
    end

    @testset "Visual" begin
        params = LSM_Params(4,2)
        lsm = LSM(params, visual=true)
        x = [1.0, 2.0, 5.0, 3.0]
        n = 3

        for _ in 1:n
            lsm(x)
        end

        @test size(lsm.states_dict["spike"])[1] == n
    end

    @testset "Constructor" begin
        params = LSM_Params(4,2)
        readout = Chain(Dense(params.ne, params.res_out, relu),
            Dense(params.res_out, params.n_out))
        lsm = LSM(params, readout)
        x = [1.0, 2.0, 5.0, 3.0]
        y = lsm(x)

        LSM(params, genPositive, readout)

        @test isa(y, Vector{Float64})
    end

    @testset "Overload" begin
        params = LSM_Params(4,2)
        lsm = LSM(params)

        @test Flux.trainable(lsm) == (lsm.readout,)
        @test CUDA.device(lsm) == Val(:cpu)
    end
end
