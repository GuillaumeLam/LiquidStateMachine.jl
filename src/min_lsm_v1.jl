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

function (res::Reservoir)(x)::Vector{Float64}
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

function (res::Reservoir)(x::Matrix{Float64})::Matrix{Float64}
	out_spike_train = []

	ignore_derivatives() do 
		readout_matrix = hcat([res(x[:,col]) for col in axes(x)[2]]...)
		out_spike_train = readout_matrix
	end

	return out_spike_train
end

# struct LSM
# 	lsm :: Chain
# end

# LSM(reservoir::Reservoir, readout::Chain) = LSM(Chain(reservoir, readout))
# (lsm::LSM)(x::Vector{Float64}) = lsm.lsm(x)

# vin = 0.8 .+ 0.4*randn(N)

function ChainRulesCore.rrule(::typeof(reservoir), x)
	y = reservoir(x)
	function pb(ȳ) 
		return ChainRulesCore.NoTangent(), bar(ȳ)
	end
	return y, pb
end

# lsm = LSM(reservoir, readout)

function multiplex(n)
	multiplexed_bits = rand(0:1, 4, n)

	vector_ones = fill(1.0, 250)
	vector_zeros = fill(0.0, 250)

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

function col_xor(col)
	channels = [mean(col[1:250]), mean(col[251:500]), mean(col[501:750]), mean(col[751:1000])]
	if sum(channels) == 1
		return [0.0, 1.0]
	else
		return [1.0, 0.0]
	end
end

# function loss(x, y)
# 	z = reservoir(x)
# 	ŷ = readout(z)
# 	Flux.mse(ŷ, y)
# end

# loss(float.(input_data[:,1]), float.(target_data[:,1]))

# loss_t = []

function tune_readout()

	test_x = multiplex(30)
	test_y = mapslices(col_xor, test_x, dims=1)

	for i in 1:10

		pre_w = copy(Flux.params(readout)[1])

		n_of_examples = 100
		input_data = multiplex(n_of_examples)
		target_data = mapslices(col_xor, input_data, dims=1)
		println("Running epoch $i")
		@time Flux.train!(lsm, [(input_data, target_data)], opt) do m, x, y
			Flux.mse(m(x), y)			
		end
		# flux_train(input_data, target_data)

		ŷ = lsm(test_x)
		mse = Flux.mse(ŷ, test_y)
		append!(mse_t, mse)
		println("Current mse: $mse")
		println("MSE hist: $mse_t")

		post_w = copy(Flux.params(readout)[1])

		Δw = post_w - pre_w

		display(heatmap(Δw))
		println(Δw)
		println(sum(Δw))

		# append!(loss_t, mean([loss(float.(input_data[:,j]), float.(target_data[:,j])) for j in 36:53]))
		# print(loss_t)
	end
end

# function  Flux.Optimise.update!(opt, xs::Zygote.Params, gs::Zygote.Params)
# 	for (x, g) in zip(xs, gs)
# 		update!(opt, x, g)
# 	end
# end

# function Flux.Optimise.update!(opt, xs::Tuple, gs)
#     for (x, g) in zip(xs, gs)
#         update!(opt, x, g)
#     end
# end

# function flux_train(x, y)
# 	grads = gradient(()-> Flux.mse(read_out(reservoir(x)), y), Θ)

# 	println("Gradients:")
# 	println(keys(grads))

# 	update!(opt, Θ, grads)
# end

reservoir_size = 1000
readout_input_width = 50

readout_output_width = 2 # binary classification
hidden_width = 7

# Make all vars float64
layers = Chain(Dense(readout_input_width, hidden_width, sigmoid), Dense(hidden_width, readout_output_width, sigmoid))
readout = Chain(layers, softmax)

# W_in = randn(Float64, readout_input_width, hidden_width)
# b_in = randn(Float64, 1, hidden_width)
# W_out = randn(Float64, hidden_width, readout_output_width)
# b_out = randn(Float64, 1, readout_output_width)

# function read_out(x::Matrix{Float64})
# 	z = σ.(x * W_in .+ b_in)
# 	y = σ.(z * W_out .+ b_out)

# 	y_sftmx = softmax(y)
# 	return y_sftmx
# end

# θ = (W_in, b_in, W_out, b_out)

# read_out(multiplex(readout_input_width))

reservoir = Reservoir(reservoir_size, readout_input_width)

lsm = Chain(reservoir, readout)

θ = Flux.params(lsm)

opt = Flux.setup(ADAM(0.04), lsm)

mse_t = []
@time tune_readout()

tune_readout()

plot(mse_t)
