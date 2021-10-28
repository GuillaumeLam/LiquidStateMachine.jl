mutable struct SpikeTrainGenerator
    d
    sample
    rng

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
        return f
    else
        stg.sample = rand(ds)
        f(_) = rand!(ds, stg.sample)
        return f
    end
end

function (stg::SpikeTrainGenerator)(m::AbstractMatrix)
    return mapslices(stg, m, dims=1)
end
