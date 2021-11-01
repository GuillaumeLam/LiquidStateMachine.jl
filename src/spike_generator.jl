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
