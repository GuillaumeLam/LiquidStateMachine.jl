mutable struct SpikeTrainGenerator
    d
    sample
    rng
    generator

    SpikeTrainGenerator(d) = new(d,[])
    SpikeTrainGenerator(d, rng) = new(d,[],rng)
end

function (stg::SpikeTrainGenerator)(x::AbstractVector)

    if all(x.==0)
        ds = product_distribution(stg.d.(x))
    else
        norm = normalize(x)
        ds = product_distribution(stg.d.(norm))
    end

    if !isnothing(stg.rng)
        stg.sample = rand(stg.rng, ds)
        f(_) =  rand!(stg.rng, ds, stg.sample)
        stg.generator = f
    else
        stg.sample = rand(ds)
        g(_) = rand!(ds, stg.sample)
        stg.generator = g
    end

    return stg.generator
end

function (stg::SpikeTrainGenerator)(m::AbstractMatrix)
    return mapslices(stg, m, dims=1)
end


mutable struct SpikeTrainDecipher
    f

    SpikeTrainDecipher(f) = new(f)
    function SpikeTrainDecipher()
        f(x; sim_τ=0.001, sim_T=0.1) = begin
            output_smmd = sum(x, dims=2)

            if all(output_smmd.==0)
                output_smmd = vec(output_smmd)
            else
                output_smmd = vec(LinearAlgebra.normalize(output_smmd, sim_T/sim_τ))
            end

            output_smmd
        end
        new(f)
    end
end

function (std::SpikeTrainDecipher)(st; sim_τ=0.001, sim_T=0.1)
    std.f(st; sim_τ=0.001, sim_T=0.1)
end
