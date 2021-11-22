mutable struct SpikeTrainGenerator
    d
    sample
    rng
    generator

    SpikeTrainGenerator(d) = new(d,[],nothing)
    SpikeTrainGenerator(d, rng) = new(d,[],rng)
end


function (stg::SpikeTrainGenerator)(x̃::AbstractVector)

    if all(x̃.==0)
        ds = product_distribution(stg.d.(x̃))
    else
        norm = normalize(x̃)
        ds = product_distribution(stg.d.(norm))
    end

    if isnothing(stg.rng)
        stg.sample = rand(ds)
        g(_) = rand!(ds, stg.sample)
        stg.generator = g
    else
        stg.sample = rand(stg.rng, ds)
        f(_) =  rand!(stg.rng, ds, stg.sample)
        stg.generator = f
    end

    return stg.generator
end

function (stg::SpikeTrainGenerator)(X̃::AbstractMatrix)
    return mapslices(stg, X̃, dims=1)
end


mutable struct SpikeTrainDecipher
    f

    SpikeTrainDecipher(f) = new(f)
    function SpikeTrainDecipher()
        f(st; sim_τ=0.001, sim_T=0.1) = begin
            output_smmd = sum(st, dims=2)

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
