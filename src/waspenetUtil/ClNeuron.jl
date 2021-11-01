struct ClNeuron{N<:AbstractNeuron, T<:Number} <: AbstractNeuron
    inner_neuron::N

    t_sls::T = 0    # Time since last spike

    c::T = 5.
    cθ::T = 5.
    cδ::T = 3.
end

function update(neuron::ClNeuron, input_update, dt, t)
    inner_output, return_neuron = update(neuron.inner_neuron, input_update, dt, t)

    # Update c based on calcium dynamics

    if inner_output == 1
        neuron.t_sls = 0
    else
        neuron.t_sls += 1
    end

    return (inner_output, ClNeuron(return_neuron))
end

function get_neuron_outputs(n::ClNeuron)
    return get_neuron_outputs(n.inner_neuron)
end

function get_neuron_states(n::ClNeuron)
    return get_neuron_states(n.inner_neuron)
end

function get_neuron_count(n::ClNeuron)
    return get_neuron_count(n.inner_neuron)
end

# find firing timing of pre and post pairs
# => check when neuron has fired timing since prev firings?
