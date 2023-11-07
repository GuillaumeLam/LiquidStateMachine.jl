# from https://github.com/anglyan/spikingnet/blob/master/julia/spnet.jl

# PROS:
# extremely simple code
# CONS:
# no GPU integration
# update rules far from lif struct

using Pkg

Pkg.activate("LSM-v1")

using Flux
using ChainRulesCore
using Plots
using SparseArrays
using Statistics
# using Zygote

mutable struct SpikingLIF
	tau :: Float64
	tref :: Float64
	v :: Float64
	v0 :: Float64
	tlast :: Float64
	inref :: Bool
end

SpikingLIF(tau :: Float64, tref:: Float64, v0 :: Float64) =
	SpikingLIF(tau, tref, 0.0, v0, 0.0, false)

struct Spike
	t :: Float64
	w :: Float64
end

const SpikeTrain = Vector{Spike}

const FanOut = Vector{Int}

mutable struct SpNet
	neurons :: Vector{SpikingLIF}
	fanout :: Vector{FanOut}
	W :: Matrix{Float64}
	td :: Float64
end

mutable struct SpikeSim
	N :: Int
	spikes :: Vector{SpikeTrain}
	time :: Float64
	dt :: Float64
end

SpikeSim(N::Int, dt::Float64) =
	SpikeSim(N, [SpikeTrain() for i=1:N], 0.0, dt)

function next_step!(spn::SpNet, spsim::SpikeSim, vin::Vector{Float64})
	spiking = []
	spout = zeros(spsim.N)
	for i=1:spsim.N
		spout[i] = next_step!(spn.neurons[i], spsim.time, spsim.dt, vin[i],
			spsim.spikes[i])
		if spout[i] > 0.5
			push!(spiking, i)
		end
	end
	stime = spsim.time + spn.td
	for isp in spiking
		for j in spn.fanout[isp]
			Wji = spn.W[j,isp]
			push!(spsim.spikes[j], Spike(stime, Wji))
		end
	end
	spsim.time += spsim.dt
	return spout
end

function next_step!(sn::SpikingLIF, time::Float64, dt::Float64,
	vext::Float64, spt::SpikeTrain)

	vne = 0.0
	while length(spt) > 0 && spt[1].t < time + dt
		spike = popfirst!(spt)
		vne += spike.w
	end
	return next_step!(sn, dt, vext, vne)
end

function next_step!(sn::SpikingLIF, dt::Float64, vin::Float64, vne::Float64)
	if sn.inref
		if sn.tlast >= sn.tref
			sn.tlast = 0.0
			sn.inref = false
		else
			sn.tlast += dt
		end
		return 0
	else
		sn.v = (sn.tau*sn.v + vin*dt + vne)/(sn.tau + dt)
		if sn.v >= sn.v0
			sn.v = 0
			sn.inref = true
			sn.tlast = 0.0
			return 1
		else
			return 0
		end
	end
end

function create_random(N::Int, p::Float64)
	flist = [FanOut() for i=1:N]
	for i = 1:N
		for j=1:N
			if i == j
				continue
			else
				if rand() < p
					push!(flist[i],j)
				end
			end
		end
	end
	return flist
end

# nlist = [SpikingLIF(8.0, 1.0, 1.0) for i=1:N]
# snet = SpNet(nlist, cm, Wnn, 1.00)
# spsim = SpikeSim(N, 0.01)
# vin = 0.8 .+ 0.4*randn(N)

struct Reservoir
	N :: Int
	nlist :: Vector{SpikingLIF}
	snet :: SpNet
	spsim :: SpikeSim
	readout_neurons :: Vector{Int}
	grad_dummy :: Float64
end

function Reservoir(N::Int, out_n_neurons::Int)
	cm = create_random(N, 0.05)
	Wnn = zeros(N,N)
	for i=1:N
		for j in cm[i]
			Wnn[j,i] = 0.2
		end
	end

	nlist = [SpikingLIF(8.0, 1.0, 1.0) for i=1:N]
	
	Reservoir(N, nlist, SpNet(nlist, cm, Wnn, 1.00), SpikeSim(N, 0.01), rand(1:N, out_n_neurons), 0.0)
end

function (res::Reservoir)(x::Vector{Float64})
	# vin = 0.8 .+ 0.4*randn(N)
	activity = []
	out = []

	@time for i=1:10000
		ignore_derivatives() do
			out = next_step!(res.snet, res.spsim, x)
			act = [j for j=1:res.N if out[j] > 0.5]
			for a in act
				push!(activity, (i,a))
			end
		end
	end

	times = [t for (t, n) in activity]
	neurons = [n for (t, n) in activity]

	sparse_activity = sparse(neurons, times, 1)
	padded_activity = hcat(sparse_activity, sparse(zeros(Int, size(sparse_activity, 1), 10000 - size(sparse_activity, 2))))

	readout_spike_train = [padded_activity[:,i] for i in res.readout_neurons]

	readout_input = sum.(readout_spike_train)

	return readout_input
end

# res(x::Matrix{Float64}) i want this function to use the res(::Vector{Float64}) function for each row of x and regroup into a matrix of size i x 2 
function (res::Reservoir)(x::Matrix{Float64})
	readout_input = [res(x[i,:]) for i in 1:size(x,1)]
	readout_input = hcat(readout_input...)
	return readout_input
end

# activity = []
# out = []

# @time for i=1:10000
#     out = next_step!(snet, spsim, vin)
#     act = [j for j=1:N if out[j] > 0.5]
#     for a in act
# #        println("$i $a")
#         push!(activity, (i,a))
#     end
# end
# println("$(length(activity))")

# times = [t for (t, n) in activity]
# neurons = [n for (t, n) in activity]

# sparse_activity = sparse(neurons, times, 1)
# padded_activity = hcat(sparse_activity, sparse(zeros(Int, size(sparse_activity, 1), 10000 - size(sparse_activity, 2))))

# println(padded_activity)

# scatter(times, neurons, markersize=1, xlabel="Time", ylabel="Neuron index", legend=false)

# readout_input_width = 20 

# readout_neurons = rand(1:N, readout_input_width)
# readout_input = [padded_activity[:,i] for i in readout_neurons]

# summed_activity = sum.(readout_input)

reservoir_size = 1000
readout_input_width = 20

readout_output_width = 1 # binary classification
readout = Chain(Dense(readout_input_width, readout_output_width, relu), softmax)

reservoir = Reservoir(reservoir_size, readout_input_width)

function ChainRulesCore.rrule(::typeof(reservoir), x)
	y = reservoir(x)
	function pb(ȳ) 
		return ChainRulesCore.NoTangent(), bar(ȳ)
	end
	return y, pb
end

# function ChainRulesCore.rrule(reservoir::Reservoir, x)
#     bar_pullback(Δy) = Tangent{Reservoir}(;grad_dummy=Δy), Δy, Δy
#     return reservoir(x), bar_pullback
# end

# z = res(vin)
# y = readout(z)

# lsm = Chain(reservoir, readout)

# lsm(x) = readout(res(x))
# Flux.@functor CustomModel
# lsm = CustomModel(lsm)

# function xor_fn(x)
#     num_categories = 4
#     y = zeros(1)
#     if x[1:250] != x[2]
#         y[1] = x[1]
#         y[2] = x[2]
#     end
#     return y
# end

function multiplex(n)
	multiplexed_bits = rand(0:1, 4, n)

	vector_ones = fill(1, 250)
	vector_zeros = fill(0, 250)

	function int_to_vec(b)
		if b == 1
			return vector_ones
		else
			return vector_zeros
		end
	end

	function col_convert(col)
		v = vcat(int_to_vec.(col)...)
		return v
	end

	res_compatible_in_data = mapslices(col_convert, multiplexed_bits, dims=1)
	return res_compatible_in_data
end

n_of_examples = 100
# input_HI_signal = rand(0:4, n_of_examples)
input_data = multiplex(n_of_examples)

function col_xor(col)
	channels = [mean(col[1:250]), mean(col[251:500]), mean(col[501:750]), mean(col[751:1000])]
	if sum(channels) == 1
		return 1
	else
		return 0
	end
end

target_data = mapslices(col_xor, input_data, dims=1)

# input_data = rand(100, reservoir_size) # 10 samples of 20-dimensional input data
# output_data = rand(100, 2)

function loss(x, y)
	z = reservoir(x)
	ŷ = readout(z)
	Flux.mse(ŷ, y)
end
# gradient(reservoir, input_data[1,:])
# rrule(reservoir, float.(input_data[:,1]))
opt = ADAM(0.01)

loss(float.(input_data[:,1]), float.(target_data[:,1]))

loss_t = []

function tune_readout()
	for i in 1:10
		println("Running epoch $i")
		for j in 1:size(input_data, 2)
			println("Running sample $j")
			@time Flux.train!(loss, Flux.params(readout), [(float.(input_data[:,j]), float.(target_data[:,j]))], opt)
		end

		append!(loss_t, mean([loss(float.(input_data[:,j]), float.(target_data[:,j])) for j in 36:53]))
	end
end

@time tune_readout()

plot(loss_t)

# for i in 1:100
#     println("Running epoch $i")
#     @time Flux.train!(loss, Flux.params(readout), [(input_data, output_data)], opt)
# end

