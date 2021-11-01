mutable struct LSM{N<:AbstractNetwork}
    readout
    reservoir::N
    preprocessing::Function
    st_gen::SpikeTrainGenerator

    states_dict

    function (lsm::LSM)(x)
        Zygote.ignore() do
            if !isnothing(lsm.states_dict)
                lsm.states_dict["env"]=[lsm.states_dict["env"];x'] # x -> dim (4,)
            end

        end

        h = Zygote.ignore() do
            x̃ = lsm.preprocessing(x)
            st = lsm.st_gen(x̃)
            return lsm.reservoir(st; visual=lsm.states_dict)
        end

        z = lsm.readout(h)

        Zygote.ignore() do
            if !isnothing(lsm.states_dict)
                lsm.states_dict["out"]=[lsm.states_dict["out"];z'] # z -> dim (2,)
            end
        end

        return z
    end

    LSM(readout, res::N, func; rng, visual) where {N<:AbstractNetwork} =
        new{N}(
            readout,
            res,
            func,
            SpikeTrainGenerator(Distributions.Bernoulli, rng),
            visual ?
                Dict(
                    "env"=>Array{Float64}(undef, 0, 4),
                    "out"=>Array{Float64}(undef, 0, 2),
                    "spike"=>Vector{Float64}(undef, 0)
                ) :
                nothing
        )
end


function LSM(params::P, readout, func::F; rng::R=StableRNGs.StableRNG(123), visual=false) where {F<:Function,P<:LSM_Params,R<:AbstractRNG}
    reservoir = init_res(params, rng)
    return LSM(readout, reservoir, func; rng=rng, visual=visual)
end

function LSM(params::P, func::F; rng::R=StableRNGs.StableRNG(123), visual=false) where {F<:Function,P<:LSM_Params,R<:AbstractRNG}
    reservoir = init_res(params, rng)
    readout = Chain(Dense(rand(rng,params.res_out, params.ne), rand(rng, params.res_out), relu),
        Dense(rand(rng, params.n_out, params.res_out), rand(rng, params.n_out)))
    return LSM(readout, reservoir, func; rng=rng, visual=visual)
end

LSM(params::P, readout; rng::R=StableRNGs.StableRNG(123), visual=false) where {P<:LSM_Params,R<:AbstractRNG} =
    LSM(params::P, readout, identity; rng=rng, visual=visual)

LSM(params::P; rng::R=StableRNGs.StableRNG(123), visual=false) where {P<:LSM_Params,R<:AbstractRNG} =
    LSM(params, identity; rng=rng, visual=visual)

###
# Overloaded functions
###

function (res::AbstractNetwork)(spike_train_generator, sim_τ=0.001, sim_T=0.1; visual)
    sim = simulate!(res, spike_train_generator, sim_τ, sim_T)

    # println(sim) => when res is learning, showing raster will be worth
    # println(size(sim.outputs)) # -> dim (182,101)
    # println(size(sim.times))

    if !isnothing(visual)
        visual["spike"] = cat(visual["spike"], [sim.outputs], dims=1)
    end

    idx = length(last(res.prev_outputs))-1

    output_smmd = sum(sim.outputs[end-idx:end-Int(0.2*(idx+1)),:],dims=2)

    if all(output_smmd.==0)
        return vec(output_smmd)
    else
        return vec(LinearAlgebra.normalize(output_smmd, sim_T/sim_τ))
    end
end

function (res::AbstractNetwork)(m::AbstractMatrix; visual)
    v = map(x -> res(x; visual=visual), vec(m))
    return hcat(v...)
end

Flux.trainable(lsm::LSM) = (lsm.readout,)

CUDA.device(lsm::LSM) = Val(:cpu)


###
# Network Constructor
###

function init_res(params::LSM_Params, rng::AbstractRNG)
    lif_params = (20., 10., 0.5, 0., 0., 0., 0.)

    ### liquid-in layer creation
    in_n = [WaspNet.LIF(lif_params...) for _ in 1:params.res_in]
    in_w = randn(rng, params.res_in, params.n_in)
    in_l = Layer(in_n, in_w)

    #=
        Maybe seperating the exc. and inh. pools will result in quicker runtime
        with similar acc
    =#

    ### res layer creation
    res_n = Vector{AbstractNeuron}([WaspNet.LIF(lif_params...) for _ in 1:params.ne])
    append!(res_n, [WaspNet.InhibNeuron(WaspNet.LIF(lif_params...)) for _ in 1:params.ni])

    W_in = cat(create_conn.(rand(rng, params.res_in, params.ne),params.K,params.PE_UB,params.res_in), zeros(params.res_in,params.ni), dims=2)'

    W_EI = create_conn.(rand(rng, params.ne,params.ni), params.C, params.EI_UB, params.ne)
    W_IE = create_conn.(rand(rng, params.ni,params.ne), params.C, params.IE_UB, params.ne)
    W_EE = W_EI*W_IE
    W_EE[diagind(W_EE)] .= 0.
    W_II = W_IE*W_EI

    W_res = cat(cat(W_EE, W_EI, dims=2),cat(W_IE, W_II, dims=2), dims=1)

    res_w = [W_in, W_res]
    conns = [1, 2]

    res = Layer(res_n, res_w, conns)

    res = Network([in_l, res], params.n_in)

    return res
end
